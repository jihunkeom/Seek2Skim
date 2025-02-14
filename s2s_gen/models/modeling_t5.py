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

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions, 
    Seq2SeqLMOutput
)
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


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
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
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.num_instances = 0
        self.all_flops = 0.
        self.encoder_flops = 0.
        self.decoder_flops = 0.

    def reset_ee_stats(self):
        self.num_instances = 0
        self.all_flops = 0.
        self.encoder_flops = 0.
        self.decoder_flops = 0.

#     def greedy_search(
#         self,
#         input_ids: torch.LongTensor,
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         max_length: Optional[int] = None,
#         pad_token_id: Optional[int] = None,
#         eos_token_id: Optional[int] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         output_scores: Optional[bool] = None,
#         return_dict_in_generate: Optional[bool] = None,
#         synced_gpus: Optional[bool] = False,
#         **model_kwargs,
#     ):
#         # init values
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#         if max_length is not None:
#             warnings.warn(
#                 "`max_length` is deprecated in this function, use"
#                 " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
#                 UserWarning,
#             )
#             stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
#         pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
#         output_scores = output_scores if output_scores is not None else self.config.output_scores
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict_in_generate = (
#             return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
#         )

#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )

#         ###
#         self.num_instances += 1
#         encoder_seq_len = torch.mean(torch.sum((model_kwargs["attention_mask"] > 0).to(torch.float32), dim=1)).item()
#         encoder_mac = compute_encoder_macs([encoder_seq_len for _ in range(self.config.num_layers)], self.config.d_model)
#         decoder_mac = 0.
#         ###

#         # keep track of which sequences are already finished
#         unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

#         this_peer_finished = False  # used by synced_gpus only
#         while True:
#             if synced_gpus:
#                 # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
#                 # The following logic allows an early break if all peers finished generating their sequence
#                 this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
#                 # send 0.0 if we finished, 1.0 otherwise
#                 dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
#                 # did all peers finish? the reduced sum will be 0.0 then
#                 if this_peer_finished_flag.item() == 0.0:
#                     break

#             # prepare model inputs
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

#             # forward pass to get next token
#             outputs = self(
#                 **model_inputs,
#                 return_dict=True,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )

#             ###
#             current_step = outputs.past_key_values[0][0].shape[2]
#             decoder_mac_ = compute_decoder_macs(
#                 step=current_step,
#                 vocab_size=self.config.vocab_size,
#                 num_layers=self.config.num_decoder_layers,
#                 enc_sentence_lengths=[encoder_seq_len for _ in range(self.config.num_layers)],
#                 dec_sentence_lengths=[current_step for _ in range(self.config.num_decoder_layers)],
#                 dim=self.config.d_model,
#             )
#             decoder_mac += decoder_mac_
#             ###

#             if synced_gpus and this_peer_finished:
#                 continue  # don't waste resources running the code we don't need

#             next_token_logits = outputs.logits[:, -1, :]

#             # pre-process distribution
#             next_tokens_scores = logits_processor(input_ids, next_token_logits)

#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (next_tokens_scores,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)

#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )

#             # argmax
#             next_tokens = torch.argmax(next_tokens_scores, dim=-1)

#             # finished sentences should have their next token be a padding token
#             if eos_token_id is not None:
#                 if pad_token_id is None:
#                     raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
#                 next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

#             # update generated ids, model inputs, and length for next step
#             input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#             )

#             # if eos_token was found in one sentence, set sentence to finished
#             if eos_token_id is not None:
#                 unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

#             # stop when each sentence is finished, or if we exceed the maximum length
#             if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True

#         ###
#         self.encoder_flops += encoder_mac
#         self.decoder_flops += decoder_mac
#         self.all_flops += (encoder_mac + decoder_mac)
#         ###

#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return GreedySearchEncoderDecoderOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                 )
#             else:
#                 return GreedySearchDecoderOnlyOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                 )
#         else:
#             return input_ids
        
def compute_encoder_macs(sentence_lengths, dim):
    def _layer_mac(seq_len, dim):
        mac = 2 * dim * (seq_len ** 2)
        mac += 12 * (dim ** 2) * seq_len
        return mac
    
    mac = 0
    for i in range(len(sentence_lengths)):
        seq_len = sentence_lengths[i]
        mac += _layer_mac(seq_len, dim)
        
    return mac

def compute_decoder_macs(step, vocab_size, num_layers, enc_sentence_lengths, dec_sentence_lengths, dim):
    def _layer_mac(step, enc_seq_len, dec_seq_len, dim):
        self_attn_mac = 2 * dec_seq_len * dim
        self_attn_mac += 4 * (dim ** 2)
        if step == 1:
            cross_attn_mac = 2 * enc_seq_len * (dim ** 2)
        else:
            cross_attn_mac = 0
        cross_attn_mac += 2 * (dim ** 2)
        cross_attn_mac += 2 * dim * enc_seq_len
        ffn_mac = 8 * (dim ** 2)
        return self_attn_mac + cross_attn_mac + ffn_mac
    
    mac = 0
    for i in range(num_layers):
        enc_seq_len = enc_sentence_lengths[i]
        dec_seq_len = dec_sentence_lengths[i]
        mac += _layer_mac(step, enc_seq_len, dec_seq_len, dim)
    mac += (dim * vocab_size)
    
    return mac