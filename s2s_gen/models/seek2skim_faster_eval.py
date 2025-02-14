"""
T5: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L19
"""
from typing import Optional, Tuple, Union

import os
import copy
import math
import datetime
import warnings
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers.models.t5.modeling_t5 import (
    T5LayerNorm,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention, 
    T5LayerFF,
    T5Block, 
    T5Stack, 
    T5ForConditionalGeneration
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.utils import logging, ModelOutput

logger = logging.get_logger(__name__)
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
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
            nn.Linear(config.d_model, config.d_model // 2),
            T5LayerNorm(config.d_model // 2, eps=config.layer_norm_epsilon),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
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
    
class Seek2SkimT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        skim_mask=None,
        decoder_step=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            # real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
            real_seq_length = decoder_step

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None

        if (not self.training) and (self.is_decoder) and (key_value_states is not None) and (skim_mask is not None):
            # skim_mask = skim_mask.to(torch.int)
            selected_indices = skim_mask[:, -1, :].nonzero(as_tuple=True)[1]
            key_states = trunc_with_mask(key_states, selected_indices, 2)
            value_states = trunc_with_mask(value_states, selected_indices, 2)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if self.is_decoder and key_value_states is not None:
            cross_attn_scores = scores.clone()

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        
        if self.training and skim_mask is not None:
            if skim_mask.dim() == 2:
                attn_weights = attn_weights * skim_mask[:, None, None, :]
            else:
                attn_weights = attn_weights * skim_mask[:, None, :, :]

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        if self.is_decoder and key_value_states is not None:
            outputs = outputs + (cross_attn_scores,)

        return outputs
    
class Seek2SkimT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = Seek2SkimT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        skim_mask=None,
        decoder_step=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skim_mask=skim_mask,
            decoder_step=decoder_step,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs
    
class Seek2SkimT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = Seek2SkimT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        skim_mask=None,
        decoder_step=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            skim_mask=skim_mask,
            decoder_step=decoder_step,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs
    
class Seek2SkimT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(Seek2SkimT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(Seek2SkimT5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        skim_mask=None,
        cross_skim_mask=None,
        decoder_step=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            # expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            # if len(past_key_value) != expected_num_past_key_values:
            #     raise ValueError(
            #         f"There should be {expected_num_past_key_values} past states. "
            #         f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
            #         f"Got {len(past_key_value)} past key / value states"
            #     )
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
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skim_mask=skim_mask,
            decoder_step=decoder_step,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

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
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
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

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
            # if present_key_value_state is not None and cross_attention_outputs[1] is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class Seek2SkimT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [Seek2SkimT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        if not self.is_decoder:
            self.self_skimmer = nn.ModuleList(
                [SelfSkimmer(config) for _ in range(config.num_layers - 1)]
            )
        else:
            self.num_skimmers = config.num_decoder_layers - config.decoder_start_skim_layer if config.decoder_first_N is None else config.decoder_first_N - config.decoder_start_skim_layer
            self.self_skimmer = nn.ModuleList(
                [SelfSkimmer(config) for _ in range(self.num_skimmers)]
            )
            self.cross_skimmer = nn.ModuleList(
                [CrossSkimmer() for _ in range(config.num_decoder_layers - 1)]
            )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self.past_skim_mask = [None for _ in range(self.num_skimmers)] if self.is_decoder else None
        self.step = 0 if self.is_decoder else None
        

    def reset_instance(self):
        self.past_skim_mask = [None for _ in range(self.num_skimmers)] if self.is_decoder else None
        self.step = 0 if self.is_decoder else None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            # past_key_values = [None] * len(self.block)
            past_key_values = (None,) * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
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
            if not self.is_decoder and self.config.encoder_first_N is not None:
                if i == self.config.encoder_first_N:
                    break
            elif self.is_decoder and self.config.decoder_first_N is not None:
                if i == self.config.decoder_first_N:
                    break

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not self.is_decoder and i > 0:
                if self.training:
                    skim_mask = nn.functional.gumbel_softmax(self.self_skimmer[i-1](hidden_states), hard=True, tau=self.config.gumbel_softmax_tau)
                    skim_mask = skim_mask[:, :, 1]
                else:
                    skim_mask = torch.argmax(self.self_skimmer[i-1](hidden_states), dim=-1)
                    if torch.sum(skim_mask, dim=-1) < 1:
                        break

                if all_skim_mask and self.training:
                    skim_mask = skim_mask * all_skim_mask[-1]
                if skim_mask is not None:
                    all_skim_mask = all_skim_mask + (skim_mask,)

                if not self.training:
                    selected_indices = skim_mask.nonzero(as_tuple=True)[1]
                    hidden_states = trunc_with_mask(hidden_states, selected_indices, 1)
                    position_bias = trunc_with_mask(position_bias, selected_indices, 2)
                    position_bias = trunc_with_mask(position_bias, selected_indices, 3)
                    if forward_skim_mask is None:
                        forward_skim_mask = torch.ones_like(skim_mask).to(dtype=torch.bool)
                
            elif self.is_decoder and i > 0:
                # print(len(self.self_skimmer), len(self.past_skim_mask), self.num_skimmers)
                # print(i, self.config.decoder_start_skim_layer, i>=self.config.decoder_start_skim_layer)
                if i >= self.config.decoder_start_skim_layer:
                    current_skimmer = i - self.config.decoder_start_skim_layer
                    # print(f"Layer {i}, start_pruner_layer: {self.config.decoder_start_skim_layer}, current skimmer idx: {current_skimmer}, num skimmers: {len(self.self_skimmer)}")
                    
                    if self.training:
                        skim_mask = nn.functional.gumbel_softmax(self.self_skimmer[current_skimmer](hidden_states), hard=True, tau=self.config.gumbel_softmax_tau)
                        skim_mask = skim_mask[:, :, 1]
                    else:
                        skim_mask = torch.argmax(self.self_skimmer[current_skimmer](hidden_states), dim=-1)
                        
                        if self.past_skim_mask[current_skimmer] is not None:
                            skim_mask = torch.cat([self.past_skim_mask[current_skimmer], skim_mask], dim=1)
                        self.past_skim_mask[current_skimmer] = skim_mask
                        if skim_mask[:, -1] < 1:
                            break
                        
                    if all_skim_mask and self.training:
                        skim_mask = skim_mask * all_skim_mask[-1]
                    if skim_mask is not None:
                        all_skim_mask = all_skim_mask + (skim_mask,)

                if encoder_hidden_states is not None:
                    if not self.training and cross_attn_scores.shape[-1] < 1:
                        cross_skim_mask = None
                        encoder_hidden_states = None
                    elif self.training:
                        cross_attn_scores = torch.mean(cross_attn_scores, dim=1)
                        cross_skim_mask = F.gumbel_softmax(self.cross_skimmer[i-1](cross_attn_scores), hard=True, tau=0.1)
                        cross_skim_mask = cross_skim_mask[:, :, :, -1]
                    elif not self.training:
                        cross_attn_scores = torch.mean(cross_attn_scores, dim=1)
                        cross_skim_mask = torch.argmax(self.cross_skimmer[i-1](cross_attn_scores), dim=-1)

                    if all_cross_skim_mask and self.training:
                        cross_skim_mask = cross_skim_mask * all_cross_skim_mask[-1]
                    if cross_skim_mask is not None:
                        all_cross_skim_mask = all_cross_skim_mask + (cross_skim_mask,)

                if not self.training:
                    if i >= self.config.decoder_start_skim_layer:
                        dec_selected_indices = skim_mask.nonzero(as_tuple=True)[1]
                        position_bias = trunc_with_mask(position_bias, dec_selected_indices, 3)
                    if forward_cross_skim_mask is None:
                        forward_cross_skim_mask = torch.ones_like(cross_skim_mask).to(dtype=torch.bool)
                    if cross_skim_mask is not None:
                        # cross_skim_mask = cross_skim_mask.to(torch.int)
                        cross_selected_indices = cross_skim_mask.nonzero(as_tuple=True)[1]
                        encoder_decoder_position_bias = trunc_with_mask(encoder_decoder_position_bias, cross_selected_indices, 3)
                        forward_cross_skim_mask[forward_cross_skim_mask.clone()] = cross_skim_mask.to(dtype=torch.bool)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
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
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

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
    
class T5ForConditionalGenerationSeek2SkimFasterEval(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = Seek2SkimT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = Seek2SkimT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.num_instances = 0
        self.num_words = 0
        self.encoder_exit_layer = 0
        self.decoder_exit_layer = 0
        self.cross_exit_layer = 0
        self.encoder_layer_tokens_left = [0.] * self.config.num_layers
        self.decoder_layer_tokens_left = [0.] * self.config.num_decoder_layers
        self.cross_layer_tokens_left = [0.] * self.config.num_decoder_layers
        self.relative_flops = 0.
        self.all_flops = 0.
        self.encoder_flops = 0.
        self.decoder_flops = 0.
        

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
        self.all_flops = 0.
        self.encoder_flops = 0.
        self.decoder_flops = 0.

    def reset_instance(self):
        self.decoder.reset_instance()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputSkim]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

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

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        classification_loss = loss
        encoder_skim_loss, decoder_skim_loss, cross_skim_loss = None, None, None
        
        if self.training:
            src_tokens_length = torch.mean(torch.sum(attention_mask.to(torch.float32), dim=-1)).item()
            encoder_skim_loss, decoder_skim_loss, cross_skim_loss = 0.0, 0.0, 0.0
            num_encoder_layers = self.config.encoder_first_N if self.config.encoder_first_N is not None else self.config.num_layers
            num_decoder_layers = self.config.decoder_first_N if self.config.decoder_first_N is not None else self.config.num_decoder_layers
            # print(f"skim_mask lens: {len(decoder_outputs.skim_mask)}, {len(decoder_outputs.cross_skim_mask)}")
            for i in range(self.config.num_layers - 1):
                if i < (num_encoder_layers - 1):
                    enc_skim_mask = encoder_outputs.skim_mask[i]
                    encoder_skim_loss += torch.mean(torch.sum(enc_skim_mask, dim=-1)) / enc_skim_mask.shape[-1]
                if i < len(decoder_outputs.skim_mask):
                    dec_skim_mask = decoder_outputs.skim_mask[i]
                    decoder_skim_loss += torch.mean(torch.sum(dec_skim_mask, dim=-1)) / dec_skim_mask.shape[-1]
                if i < len(decoder_outputs.cross_skim_mask):
                    cross_skim_mask = decoder_outputs.cross_skim_mask[i]
                    cross_skim_loss += torch.mean(torch.sum(cross_skim_mask, dim=-1)) / src_tokens_length
            encoder_skim_loss /= (num_encoder_layers - 1)
            decoder_skim_loss /= len(decoder_outputs.skim_mask)
            cross_skim_loss /= len(decoder_outputs.cross_skim_mask)
            

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
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):
        ###
        # self.num_instances += 1
        self.reset_instance()
        ###

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        ###
        # encoder_seq_len = torch.mean(torch.sum((model_kwargs["attention_mask"] > 0).to(torch.float32), dim=1))    
        # encoder_exit_layer = len(model_kwargs["encoder_outputs"].skim_mask) + 1
        # self.encoder_exit_layer += encoder_exit_layer
        # self.encoder_layer_tokens_left[0] += 1.
        # for i in range(encoder_exit_layer - 1):
        #     accumulated_skim_mask = torch.mean(torch.sum(model_kwargs["encoder_outputs"].skim_mask[i].to(torch.float32), dim=-1))
        #     tokens_ratio = (accumulated_skim_mask / encoder_seq_len).item()
        #     self.encoder_layer_tokens_left[i+1] += tokens_ratio
            
        # encoder_seq_lens = [encoder_seq_len.item() for _ in range(self.config.num_layers)]
        # encoder_skimmed_lens = [encoder_seq_len.item()] + [x.sum(dim=1).item() for x in model_kwargs["encoder_outputs"].skim_mask]
        # encoder_mac = compute_encoder_macs(encoder_seq_lens, self.config.d_model, False)
        # encoder_skimmed_mac = compute_encoder_macs(encoder_skimmed_lens, self.config.d_model, True)
        # decoder_mac = 0.
        # decoder_skimmed_mac = 0.
        # enc_kv_proj = [False for _ in range(self.config.num_decoder_layers)]
        # enc_kv_proj2 = [False for _ in range(self.config.num_decoder_layers)]
        ###

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            ###
            # self.num_words += 1
            # decoder_exit_layer = len(outputs.decoder_skim_mask) + self.config.decoder_start_skim_layer
            # self.decoder_exit_layer += decoder_exit_layer
            # self.cross_exit_layer += len(outputs.cross_skim_mask) + 1

            # decoder_seq_lens = [self.decoder.step for _ in range(self.config.num_decoder_layers)]
            # decoder_skimmed_lens = [self.decoder.step for _ in range(self.config.decoder_start_skim_layer)] + [x.sum(dim=1).item() for x in self.decoder.past_skim_mask if x is not None]
            # # print(decoder_exit_layer, decoder_skimmed_lens, len(decoder_skimmed_lens))
            # cross_skimmed_lens = [encoder_seq_len.item()] + [x[:, -1, :].sum(dim=1).item() for x in outputs.cross_skim_mask]

            # self.decoder_layer_tokens_left[0] += 1
            # self.cross_layer_tokens_left[0] += 1
            # for i in range(decoder_exit_layer-1):
            #     self.decoder_layer_tokens_left[i+1] += 1
            #     if i < len(outputs.cross_skim_mask):
            #         self.cross_layer_tokens_left[i+1] += (torch.mean(torch.sum(outputs.cross_skim_mask[i].to(torch.float32), dim=-1)) / encoder_seq_len).item()
            
            # decoder_mac_, enc_kv_proj = compute_decoder_macs2(
            #     enc_kv_proj=enc_kv_proj,
            #     vocab_size=self.config.vocab_size,
            #     exit_layer=self.config.num_decoder_layers,
            #     enc_sentence_lengths=encoder_seq_lens,
            #     dec_sentence_lengths=decoder_seq_lens,
            #     dim=self.config.d_model,
            #     encoder_skimmed_lengths=None,
            #     skim=False,
            # )
            # decoder_skimmed_mac_, enc_kv_proj2 = compute_decoder_macs2(
            #     enc_kv_proj=enc_kv_proj2,
            #     vocab_size=self.config.vocab_size,
            #     exit_layer=decoder_exit_layer,
            #     enc_sentence_lengths=encoder_seq_lens,
            #     dec_sentence_lengths=decoder_skimmed_lens,
            #     dim=self.config.d_model,
            #     encoder_skimmed_lengths=cross_skimmed_lens,
            #     skim=True,
            #     decoder_start_skim_layer=self.config.decoder_start_skim_layer,
            # )
            # decoder_mac += decoder_mac_
            # decoder_skimmed_mac += decoder_skimmed_mac_
            ###
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        # self.relative_flops += (encoder_skimmed_mac + decoder_skimmed_mac) / (encoder_mac + decoder_mac)
        # self.encoder_flops += encoder_skimmed_mac
        # self.decoder_flops += decoder_skimmed_mac
        # self.all_flops += (encoder_skimmed_mac + decoder_skimmed_mac)
        # print(input_ids.shape, sum([(2 * enc_seq_len * (self.config.d_model ** 2)) for enc_seq_len in encoder_seq_lens]) / decoder_skimmed_mac)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    
def compute_encoder_macs(sentence_lengths, dim, skim=False):
    def _layer_mac(seq_len, dim):
        mac = 2 * dim * (seq_len ** 2)
        mac += 12 * (dim ** 2) * seq_len
        return mac

    def _skim_mac(seq_len, dim):
        skim_mac = (dim * (dim // 2) * seq_len) + ((dim // 2) * 2 * seq_len)
        return skim_mac

    mac = 0
    for i in range(len(sentence_lengths)):
        seq_len = sentence_lengths[i]
        mac += _layer_mac(seq_len, dim)
        if skim and i < len(sentence_lengths)-1:
            mac += _skim_mac(seq_len, dim)
    
    return mac

def compute_decoder_macs2(enc_kv_proj, vocab_size, exit_layer, enc_sentence_lengths, dec_sentence_lengths, dim, encoder_skimmed_lengths=None, skim=False, decoder_start_skim_layer=1):
    def _layer_mac(enc_skimmed_seq_len, dec_seq_len, dim):
        self_attn_mac = 2 * dec_seq_len * dim
        self_attn_mac += 4 * (dim ** 2)
        cross_attn_mac = 2 * (dim ** 2)
        cross_attn_mac += 2 * dim * enc_skimmed_seq_len
        ffn_mac = 8 * (dim ** 2)
        return self_attn_mac + cross_attn_mac + ffn_mac
    
    def _skim_mac(dim):
        return (dim * (dim // 2)) + ((dim // 2) * 2)

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
        if skim:
            if (i < exit_layer-1) and (i >= decoder_start_skim_layer):
                mac += _skim_mac(dim)
        if enc_kv_proj[i] == False:
            enc_kv_proj[i] = True
            mac += 2 * enc_seq_len * (dim ** 2)

    mac += (dim * vocab_size)

    return mac, enc_kv_proj