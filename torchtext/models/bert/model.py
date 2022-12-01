# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import (
    TransformerEncoder,
    TransformerOutput,
)
from torchmultimodal.utils.attention import get_extended_attention_mask
from torchtext.models.bert.text_embedding import BERTTextEmbeddings


class BERTTextEncoder(nn.Module):
    """
    General text transformer encoder with embeddings, following BERT.
    Can be constructed with any user-provided embeddings and encoder.

    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870

    Attributes:
        embeddings (nn.Module): Module that projects text token ids into embeddings.
            See :py:class: `torchtext.models.bert.text_embedding.BERTTextEmbeddings` for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class:
            `torchmultimodal.modules.layers.transformer.TransformerEncoder` for interface.
        layernorm (nn.Module, optional): Module for layernorm to be applied after encoder. Defaults to ``None``.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: `torchmultimodal.models.flava.transformer.init_transformer_weights`
            as an example. Defaults to ``None``.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape [batch, seq_len]
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere

    Raises:
        ValueError: if input_ids and inputs_embeds are both ``None``.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: Optional[nn.Module] = None,
        pooler: Optional[nn.Module] = None,
        weight_init_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        # TODO: could be upstreamed to TransformerEncoder?
        self.layernorm = layernorm
        self.pooler = pooler

        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids or inputs_embeds must not be None")

        # only mask out padding token if no mask specified
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            if hasattr(self.embeddings, "pad_token_id"):
                attention_mask[input_ids == self.embeddings.pad_token_id] = 0

        # massage attention mask to correct shape for transformer
        attention_mask = get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
            return_hidden_states=return_hidden_states,
        )

        last_hidden_state = encoder_output.last_hidden_state
        pooled_output = encoder_output.pooler_output
        if self.layernorm:
            last_hidden_state = self.layernorm(last_hidden_state)
        if self.pooler:
            pooled_output = self.pooler(last_hidden_state)

        return TransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


def bert_text_encoder(
    # transformer encoder params
    hidden_size: int = 768,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    dropout: float = 0.1,
    transform_act_fn: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    norm_first: bool = False,
    # text embedding params
    vocab_size: int = 30522,
    max_position_embeddings: int = 512,
    type_vocab_size: int = 2,
    pad_token_id: int = 0,
    offset_pos_ids: bool = False,
    # layernorm and pooler
    layernorm: Optional[nn.Module] = None,
    pooler: Optional[nn.Module] = None,
    weight_init_fn: Optional[Callable] = None,
) -> BERTTextEncoder:
    """
    Returns a BERTTextEncoder with default params identical to HuggingFace's ``bert-base-uncased``.
    Ref: https://huggingface.co/bert-base-uncased/resolve/main/config.json. See :py:class:
    `torchtext.models.bert.text_embedding.BERTTextEmbeddings` and :py:class:
    `torchmultimodal.modules.layers.transformer.TransformerEncoder` for details on parameters.
    """
    embeddings = BERTTextEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        offset_pos_ids=offset_pos_ids,
    )
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        dropout=dropout,
        activation=transform_act_fn,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
    )
    return BERTTextEncoder(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
        weight_init_fn=weight_init_fn,
    )
