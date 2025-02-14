""" PyTorch T5 model. """
import copy
import math
import torch
import warnings
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer, PreTrainedModel
from transformers.utils import logging
from transformers.file_utils import ModelOutput

from transformers.modeling_t5 import T5PreTrainedModel, T5LayerFF, T5LayerNorm, T5DenseReluDense
from .configuration_t5 import T5Config

logger = logging.get_logger(__name__)

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[Tuple[torch.FloatTensor]] = None
    cross_skim_mask: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Seq2SeqLMOutputSkim(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_skim_mask: Optional[Tuple[torch.FloatTensor]] = None
    decoder_skim_mask: Optional[Tuple[torch.FloatTensor]] = None
    cross_skim_mask: Optional[Tuple[torch.FloatTensor]] = None
    encoder_skim_loss: Optional[torch.FloatTensor] = None
    decoder_skim_loss: Optional[torch.FloatTensor] = None
    cross_skim_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None

def trunc_with_mask(input_, selected_indices, dim):
    trunc_input = torch.index_select(input_, dim, selected_indices)
    return trunc_input

def init_skim_predictor(module_list, mean_bias=5.0):
    for module in module_list:
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("only support initialization of linear skim predictor")
        
        module.bias.data[1].normal_(mean=mean_bias, std=0.02)
        module.bias.data[0].normal_(mean=-mean_bias, std=0.02)
        module.weight.data.normal_(mean=0.0, std=0.02)

        module._skim_initialized = True

class SelfSkimmer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.predictor = nn.Sequential(
            T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            # nn.Linear(config.d_model, config.d_model // 2),
            # T5LayerNorm(config.d_model // 2, eps=config.layer_norm_epsilon),
            nn.Linear(config.d_model, config.d_model),
            T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon),
            nn.GELU(),
            # nn.Linear(config.d_model // 2, 2),
            nn.Linear(config.d_model, 2)
        )
        init_skim_predictor([self.predictor[-1]])

    def forward(self, hidden_states):
        return self.predictor(hidden_states)
    
class CrossSkimmer(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Linear(1, 2)
        init_skim_predictor([self.predictor])
        
    def forward(self, attn_score):
        return self.predictor(attn_score.unsqueeze(-1))

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, is_bidirectional=False):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.is_bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        skim_mask=None,
        decoder_step=None,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states. {}".format(
                len(past_key_value), past_key_value
            )
            # real_qlen = qlen + past_key_value[0].shape[2] if query_length is None else query_length
            real_qlen = decoder_step
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value is not None:
            if kv is None:
                k_, v_ = past_key_value
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        if (not self.training) and (self.is_decoder) and (kv is not None) and (skim_mask is not None):
            selected_indices = skim_mask[:, -1, :].nonzero(as_tuple=True)[1]
            k = trunc_with_mask(k, selected_indices, 2)
            v = trunc_with_mask(v, selected_indices, 2)
            
        # (bs, n_heads, qlen, klen)
        scores = torch.matmul(
            q, k.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", q, k), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -qlen:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        
        if self.training and skim_mask is not None:
            if skim_mask.dim() == 2:
                weights = weights * skim_mask[:, None, None, :]
            else:
                weights = weights * skim_mask[:, None, :, :]
            
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        
        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)

        if self.is_decoder and kv is not None:
            outputs = outputs + (scores,)

        return outputs

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias,
            is_bidirectional=not config.is_decoder
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            skim_mask=None,
            decoder_step=None,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skim_mask=skim_mask,
            decoder_step=decoder_step,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=True
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        skim_mask=None,
        decoder_step=None,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            skim_mask=skim_mask,
            decoder_step=decoder_step,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
            
        self.layer.append(T5LayerFF(config))
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        skim_mask=None,
        cross_skim_mask=None,
        decoder_step=None,
    ):
        if past_key_value is not None:
            # assert self.is_decoder, "Only decoder can use `past_key_values`"
            # expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            # error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
            #     expected_num_past_key_values,
            #     "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
            #     len(past_key_value),
            # )
            # assert len(past_key_value) == expected_num_past_key_values, error_message
            if len(past_key_value) == 4:
                self_attn_past_key_value = past_key_value[:2]
                cross_attn_past_key_value = past_key_value[2:]
            elif len(past_key_value) == 2:
                self_attn_past_key_value = past_key_value[:2]
                cross_attn_past_key_value = None
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skim_mask=skim_mask,
            decoder_step=decoder_step,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            if self.training and cross_skim_mask is not None:
                self_attn_hidden_states = hidden_states.clone()
                cross_exit_skim_mask = cross_skim_mask.to(dtype=torch.bool)
                cross_exit_skim_mask = torch.all(~cross_exit_skim_mask, dim=-1, keepdim=True).to(dtype=torch.float32)

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skim_mask=cross_skim_mask,
                decoder_step=decoder_step,
            )
            hidden_states = cross_attention_outputs[0]

            if self.training and cross_skim_mask is not None:
                hidden_states = self_attn_hidden_states*cross_exit_skim_mask + hidden_states*(1-cross_exit_skim_mask)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        if self.is_decoder and not do_cross_attention and present_key_value_state is not None and cross_attn_past_key_value is not None:
            present_key_value_state = present_key_value_state + cross_attn_past_key_value
        
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)

class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0))
             for i in range(config.num_layers)]
        )
        self.self_skimmer = nn.ModuleList(
            [SelfSkimmer(config) for _ in range(config.num_layers - 1)]
        )
        if self.is_decoder:
            self.cross_skimmer = nn.ModuleList(
                [CrossSkimmer() for _ in range(config.num_layers - 1)]
            )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.init_weights()

        self.past_skim_mask = [None for _ in range(config.num_layers - 1)] if self.is_decoder else None
        self.step = 0 if self.is_decoder else None

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def reset_instance(self):
        self.past_skim_mask = [None for _ in range(self.config.num_layers - 1)] if self.is_decoder else None
        self.step = 0 if self.is_decoder else None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            # past_key_values = [None] * len(self.block)
            past_key_values = (None,) * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)

        all_skim_mask = ()
        all_cross_skim_mask = () if self.is_decoder else None
        skim_mask, cross_skim_mask = None, None
        forward_skim_mask, forward_cross_skim_mask = None, None

        if self.is_decoder and not self.training:
            self.step += 1

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:
                if self.training:
                    skim_mask = nn.functional.gumbel_softmax(self.self_skimmer[i-1](hidden_states), hard=True, tau=self.config.gumbel_softmax_tau)
                    skim_mask = skim_mask[:, :, 1]
                else:
                    skim_mask = torch.argmax(self.self_skimmer[i-1](hidden_states), dim=-1)
                    if self.is_decoder:
                        if self.past_skim_mask[i-1] is not None:
                            skim_mask = torch.cat([self.past_skim_mask[i-1], skim_mask], dim=1)
                        self.past_skim_mask[i-1] = skim_mask
                        if skim_mask[:, -1] < 1:
                            break
                    elif not self.is_decoder and torch.sum(skim_mask, dim=-1) < 1:
                        break

                if all_skim_mask and self.training:
                    skim_mask = skim_mask * all_skim_mask[-1]
                if skim_mask is not None:
                    all_skim_mask = all_skim_mask + (skim_mask,)

                if self.is_decoder and encoder_hidden_states is not None:
                    if not self.training and cross_attn_scores.shape[-1] < 1:
                        cross_skim_mask = None
                        encoder_hidden_states = None
                    elif self.training:
                        cross_attn_scores = torch.mean(cross_attn_scores, dim=1)
                        cross_skim_mask = F.gumbel_softmax(self.cross_skimmer[i-1](cross_attn_scores), hard=True, tau=self.config.gumbel_softmax_tau)
                        cross_skim_mask = cross_skim_mask[:, :, :, -1]
                    elif not self.training:
                        cross_attn_scores = torch.mean(cross_attn_scores, dim=1)
                        cross_skim_mask = torch.argmax(self.cross_skimmer[i-1](cross_attn_scores), dim=-1)

                    if all_cross_skim_mask and self.training:
                        cross_skim_mask = cross_skim_mask * all_cross_skim_mask[-1]
                    if cross_skim_mask is not None:
                        all_cross_skim_mask = all_cross_skim_mask + (cross_skim_mask,)

                if not self.training:
                    if not self.is_decoder:
                        selected_indices = skim_mask.nonzero(as_tuple=True)[1]
                        hidden_states = trunc_with_mask(hidden_states, selected_indices, 1)
                        position_bias = trunc_with_mask(position_bias, selected_indices, 2)
                        position_bias = trunc_with_mask(position_bias, selected_indices, 3)
                        if forward_skim_mask is None:
                            forward_skim_mask = torch.ones_like(skim_mask).to(dtype=torch.bool)
                    else:
                        dec_selected_indices = skim_mask.nonzero(as_tuple=True)[1]
                        position_bias = trunc_with_mask(position_bias, dec_selected_indices, 3)
                        if forward_cross_skim_mask is None:
                            forward_cross_skim_mask = torch.ones_like(cross_skim_mask).to(dtype=torch.bool)
                        if cross_skim_mask is not None:
                            cross_selected_indices = cross_skim_mask.nonzero(as_tuple=True)[1]
                            encoder_decoder_position_bias = trunc_with_mask(encoder_decoder_position_bias, cross_selected_indices, 3)
                            forward_cross_skim_mask[forward_cross_skim_mask.clone()] = cross_skim_mask.to(dtype=torch.bool)
                        ###
                        # if past_key_value is not None and len(past_key_value) == 2:
                        #     past_key_value = past_key_value + (None, None)
                        ###

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skim_mask=skim_mask,
                cross_skim_mask=cross_skim_mask if (self.training or i == 0) else forward_cross_skim_mask,
                decoder_step=self.step if self.is_decoder else None,
            )
            if self.is_decoder:
                cross_attn_scores = layer_outputs[-1]
            
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights),
            # (self-attention position bias), (cross-attention weights),
            # (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
                    
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4 if i == 0 else 3],)

            if self.training:
                if skim_mask is not None:
                    forward_hidden_states = forward_hidden_states*(1-skim_mask.view(*skim_mask.shape, 1)) + hidden_states*skim_mask.view(*skim_mask.shape, 1)
                else:
                    forward_hidden_states = hidden_states.clone()
            elif not self.training and not self.is_decoder:
                if forward_skim_mask is not None:
                    forward_skim_mask[forward_skim_mask.clone()] = skim_mask.to(dtype=torch.bool)
                    forward_hidden_states[forward_skim_mask] = hidden_states
                else:
                    forward_hidden_states = hidden_states.clone()

        if self.training or not self.is_decoder:
            hidden_states = self.final_layer_norm(forward_hidden_states)
        else:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if use_cache and len(present_key_value_states) < len(past_key_values):
            present_key_value_states = present_key_value_states + past_key_values[len(present_key_value_states):]

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentionsSkim(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            skim_mask=all_skim_mask,
            cross_skim_mask=all_cross_skim_mask,
        )
    
class T5ForConditionalGenerationSeek2SkimFasterEval(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight",
                               r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()

        self.num_instances = 0
        self.num_words = 0
        self.encoder_exit_layer = 0
        self.decoder_exit_layer = 0
        self.cross_exit_layer = 0
        self.encoder_layer_tokens_left = [0.] * self.config.num_layers
        self.decoder_layer_tokens_left = [0.] * self.config.num_decoder_layers
        self.cross_layer_tokens_left = [0.] * self.config.num_decoder_layers
        self.relative_flops = 0.
        self.encoder_saved_flops = 0.
        self.decoder_saved_flops = 0.

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def reset_stats(self):        
        self.num_instances = 0
        self.num_words = 0
        self.encoder_exit_layer = 0
        self.decoder_exit_layer = 0
        self.cross_exit_layer = 0
        self.encoder_layer_tokens_left = [0.] * self.config.num_layers
        self.decoder_layer_tokens_left = [0.] * self.config.num_decoder_layers
        self.cross_layer_tokens_left = [0.] * self.config.num_decoder_layers
        self.relative_flops = 0.
        self.encoder_saved_flops = 0.
        self.decoder_saved_flops = 0.

    def reset_instance(self):
        self.decoder.reset_instance()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
        `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored
            (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small',
            return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1>
            park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the
            <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning
            a dog is good for you ", return_tensors="pt").input_ids# Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        if "task" in kwargs:
            task = kwargs.pop("task")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutputWithPastAndCrossAttentionsSkim):
            encoder_outputs = BaseModelOutputWithPastAndCrossAttentionsSkim(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            
        classification_loss = loss
        encoder_skim_loss, decoder_skim_loss, cross_skim_loss = None, None, None
        
        src_tokens_length = torch.mean(torch.sum(attention_mask.to(torch.float32), dim=-1))
        if self.training:    
            encoder_skim_loss, decoder_skim_loss, cross_skim_loss = 0.0, 0.0, 0.0
            for i in range(self.config.num_layers - 1):
                enc_skim_mask = encoder_outputs.skim_mask[i]
                dec_skim_mask = decoder_outputs.skim_mask[i]
                cross_skim_mask = decoder_outputs.cross_skim_mask[i]
                encoder_skim_loss += torch.mean(torch.sum(enc_skim_mask, dim=-1)) / enc_skim_mask.shape[-1]
                decoder_skim_loss += torch.mean(torch.sum(dec_skim_mask, dim=-1)) / dec_skim_mask.shape[-1]
                cross_skim_loss += torch.mean(torch.sum(cross_skim_mask, dim=-1)) / src_tokens_length
            encoder_skim_loss /= (self.config.num_layers - 1)
            decoder_skim_loss /= (self.config.num_decoder_layers - 1)
            cross_skim_loss /= (self.config.num_decoder_layers - 1)

        if labels is not None:
            loss = classification_loss + (self.config.encoder_skim_factor * encoder_skim_loss) + (self.config.decoder_skim_factor * decoder_skim_loss) + (self.config.cross_skim_factor * cross_skim_loss)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputSkim(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_skim_mask=encoder_outputs.skim_mask,
            decoder_skim_mask=decoder_outputs.skim_mask,
            cross_skim_mask=decoder_outputs.cross_skim_mask,
            encoder_skim_loss=encoder_skim_loss,
            decoder_skim_loss=decoder_skim_loss,
            cross_skim_loss=cross_skim_loss,
            classification_loss=classification_loss,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
    
    def greedy_search(
            self,
            input_ids,
            logits_processor=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            **model_kwargs
    ):
        self.reset_instance()
        self.num_instances += 1
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )
        ###
        encoder_seq_len = torch.mean(torch.sum((model_kwargs["attention_mask"] > 0).to(torch.float32), dim=1))
        encoder_exit_layer = len(model_kwargs["encoder_outputs"].skim_mask) + 1
        self.encoder_exit_layer += encoder_exit_layer
        self.encoder_layer_tokens_left[0] += 1.
        for i in range(encoder_exit_layer - 1):
            accumulated_skim_mask = torch.mean(torch.sum(model_kwargs["encoder_outputs"].skim_mask[i].to(torch.float32), dim=-1))
            tokens_ratio = (accumulated_skim_mask / encoder_seq_len).item()
            self.encoder_layer_tokens_left[i+1] += tokens_ratio
        
        encoder_seq_lens = [encoder_seq_len.item() for _ in range(self.config.num_layers)]
        encoder_skimmed_lens = [encoder_seq_len.item()] + [x.sum(dim=1).item() for x in model_kwargs["encoder_outputs"].skim_mask]
        encoder_mac = compute_encoder_macs(encoder_seq_lens, self.config.d_model, False)
        encoder_skimmed_mac = compute_encoder_macs(encoder_skimmed_lens, self.config.d_model, True)
        encoder_saved_mac = encoder_skimmed_mac / encoder_mac
        self.encoder_saved_flops += encoder_saved_mac
        decoder_mac = 0.
        decoder_skimmed_mac = 0.
        enc_kv_proj = [False for _ in range(self.config.num_decoder_layers)]
        enc_kv_proj2 = [False for _ in range(self.config.num_decoder_layers)]
        # print("Encoder info")
        # print(encoder_seq_lens)
        # print(encoder_skimmed_lens, encoder_exit_layer)
        # print(f"Encoder Mac: {encoder_mac}, Encoder Skimmed Mac: {encoder_skimmed_mac}")
        # print(f"Enc KV proj: {enc_kv_proj}")
        
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            scores = logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(scores, dim=-1)

            ###
            self.num_words += 1
            decoder_exit_layer = len(outputs.decoder_skim_mask) + 1
            self.decoder_exit_layer += decoder_exit_layer
            self.cross_exit_layer += len(outputs.cross_skim_mask) + 1

            decoder_seq_lens = [self.decoder.step for _ in range(self.config.num_decoder_layers)]
            decoder_skimmed_lens = [self.decoder.step] + [x.sum(dim=1).item() for x in self.decoder.past_skim_mask if x is not None]
            cross_skimmed_lens = [encoder_seq_len.item()] + [x[:, -1, :].sum(dim=1).item() for x in outputs.cross_skim_mask]

            self.decoder_layer_tokens_left[0] += 1
            self.cross_layer_tokens_left[0] += 1
            for i in range(decoder_exit_layer-1):
                self.decoder_layer_tokens_left[i+1] += 1
                if i < len(outputs.cross_skim_mask):
                    self.cross_layer_tokens_left[i+1] += (torch.mean(torch.sum(outputs.cross_skim_mask[i].to(torch.float32), dim=-1)) / encoder_seq_len).item()
            
            decoder_mac_, enc_kv_proj = compute_decoder_macs2(
                enc_kv_proj=enc_kv_proj,
                vocab_size=self.config.vocab_size,
                exit_layer=self.config.num_decoder_layers,
                enc_sentence_lengths=encoder_seq_lens,
                dec_sentence_lengths=decoder_seq_lens,
                dim=self.config.d_model,
                encoder_skimmed_lengths=None,
                skim=False,
            )
            decoder_skimmed_mac_, enc_kv_proj2 = compute_decoder_macs2(
                enc_kv_proj=enc_kv_proj2,
                vocab_size=self.config.vocab_size,
                exit_layer=decoder_exit_layer,
                enc_sentence_lengths=encoder_seq_lens,
                dec_sentence_lengths=decoder_skimmed_lens,
                dim=self.config.d_model,
                encoder_skimmed_lengths=cross_skimmed_lens,
                skim=True,
            )
            decoder_mac += decoder_mac_
            decoder_skimmed_mac += decoder_skimmed_mac_
            # print(f"Step: {self.decoder.step}, enc_kv_proj: {enc_kv_proj}")
            # print(decoder_seq_lens)
            # print(decoder_skimmed_lens)
            # print(cross_skimmed_lens)
            # print(len(outputs.decoder_skim_mask) + 1, len(outputs.cross_skim_mask) + 1)
            # print(f"Ratio: {round(decoder_skimmed_mac_ / decoder_mac_, 2)} Decoder Mac: {decoder_mac_}, Decoder SKimmed Mac: {decoder_skimmed_mac_}")
            ###

            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined"
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if unfinished_sequences.max() == 0:
                break

            cur_len = cur_len + 1
        
        self.relative_flops += (encoder_skimmed_mac + decoder_skimmed_mac) / (encoder_mac + decoder_mac)
        self.decoder_saved_flops += decoder_skimmed_mac / decoder_mac
        # print(f"Overall Ratio: {(encoder_skimmed_mac + decoder_skimmed_mac) / (encoder_mac + decoder_mac)}, Decoder Ratio: {decoder_skimmed_mac / decoder_mac}")
        return input_ids
    
def compute_encoder_macs(sentence_lengths, dim, skim=False):
    def _layer_mac(seq_len, dim):
        mac = 2 * dim * (seq_len ** 2)
        mac += 12 * (dim ** 2) * seq_len
        return mac

    def _skim_mac(seq_len, dim):
        # skim_mac = (dim * (dim // 2) * seq_len) + ((dim // 2) * 2 * seq_len)
        skim_mac = (dim * dim * seq_len) + (dim * 2 * seq_len)
        return skim_mac

    mac = 0
    for i in range(len(sentence_lengths)):
        seq_len = sentence_lengths[i]
        mac += _layer_mac(seq_len, dim)
        if skim and i < len(sentence_lengths)-1:
            mac += _skim_mac(seq_len, dim)
    
    return mac

# def compute_decoder_macs(step, vocab_size, exit_layer, enc_sentence_lengths, dec_sentence_lengths, dim, encoder_skimmed_lengths=None, skim=False):
#     def _layer_mac(step, enc_seq_len, enc_skimmed_seq_len, dec_seq_len, dim):
#         self_attn_mac = 2 * dec_seq_len * dim
#         self_attn_mac += 4 * (dim ** 2)
#         if step == 1:
#             cross_attn_mac = 2 * enc_seq_len * (dim ** 2)
#         else:
#             cross_attn_mac = 0
#         cross_attn_mac += 2 * (dim ** 2)
#         cross_attn_mac += 2 * dim * enc_skimmed_seq_len
#         ffn_mac = 8 * (dim ** 2)
#         return self_attn_mac + cross_attn_mac + ffn_mac
    
#     def _skim_mac(dim):
#         return (dim * (dim // 2)) + ((dim // 2) * 2)

#     if len(enc_sentence_lengths) < exit_layer:
#         enc_sentence_lengths = enc_sentence_lengths + ([0.] * (exit_layer - len(enc_sentence_lengths)))
#     if encoder_skimmed_lengths is not None:
#         if len(encoder_skimmed_lengths) < exit_layer:
#             encoder_skimmed_lengths = encoder_skimmed_lengths + ([0.] * (exit_layer - len(encoder_skimmed_lengths)))

#     mac = 0
#     for i in range(exit_layer):
#         enc_seq_len = enc_sentence_lengths[i]
#         dec_seq_len = dec_sentence_lengths[i]
#         enc_skimmed_seq_len = encoder_skimmed_lengths[i] if encoder_skimmed_lengths is not None else enc_seq_len
#         mac += _layer_mac(step, enc_seq_len, enc_skimmed_seq_len, dec_seq_len, dim)
#         if skim and i < exit_layer-1:
#             mac += _skim_mac(dim)
#     mac += (dim * vocab_size)

#     return mac

def compute_decoder_macs2(enc_kv_proj, vocab_size, exit_layer, enc_sentence_lengths, dec_sentence_lengths, dim, encoder_skimmed_lengths=None, skim=False):
    def _layer_mac(enc_skimmed_seq_len, dec_seq_len, dim):
        self_attn_mac = 2 * dec_seq_len * dim
        self_attn_mac += 4 * (dim ** 2)
        cross_attn_mac = 2 * (dim ** 2)
        cross_attn_mac += 2 * dim * enc_skimmed_seq_len
        ffn_mac = 8 * (dim ** 2)
        return self_attn_mac + cross_attn_mac + ffn_mac
    
    def _skim_mac(dim):
        # return (dim * (dim // 2)) + ((dim // 2) * 2)
        return (dim * dim) + (dim * 2)

    if len(enc_sentence_lengths) < exit_layer:
        enc_sentence_lengths = enc_sentence_lengths + ([0.] * (exit_layer - len(enc_sentence_lengths)))
    if encoder_skimmed_lengths is not None:
        if len(encoder_skimmed_lengths) < exit_layer:
            encoder_skimmed_lengths = encoder_skimmed_lengths + ([0.] * (exit_layer - len(encoder_skimmed_lengths)))

    mac = 0
    for i in range(exit_layer):
        enc_seq_len = enc_sentence_lengths[i]
        dec_seq_len = dec_sentence_lengths[i]
        enc_skimmed_seq_len = encoder_skimmed_lengths[i] if encoder_skimmed_lengths is not None else enc_seq_len
        mac += _layer_mac(enc_skimmed_seq_len, dec_seq_len, dim)
        if skim and i < exit_layer-1:
            mac += _skim_mac(dim)
        if enc_kv_proj[i] == False:
            enc_kv_proj[i] = True
            mac += 2 * enc_seq_len * (dim ** 2)

    mac += (dim * vocab_size)

    return mac, enc_kv_proj