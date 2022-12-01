# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor


class BERTTextEmbeddings(nn.Module):
    """Construct word, position, and token type embeddings following BERT, similar to HuggingFace BertEmbeddings

    Attributes:
        hidden_size (int): size of embedding space. Default is 768.
        vocab_size (int): size of vocabulary. Default is 30522.
        pad_token_id (int): id used for padding token. Default is 0.
        max_position_embeddings (int): the highest position id number, or max sequence length. Default is 512.
        type_vocab_size (int): the highest token type id number. Default is 2.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        dropout (float): dropout probability after all embeddings and layernorm
        offset_pos_ids (bool): if True, shift position ids by one for the padding token. Used in RoBERTa.
            Default is False.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        offset_pos_ids: bool = False,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id
        self.offset_pos_ids = offset_pos_ids

    def create_position_ids_from_input_ids(self, input_ids: Tensor) -> Tensor:
        """
        Replace non-padding symbols with their position numbers.
        Position numbers begin at pad_token_id+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Inputs: input_ids (Tensor): Tensor from which to create position IDs.
                pad_token_id (int): Padding index
                    (determines starting point of position IDs).
        """
        mask = input_ids.ne(self.pad_token_id).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + self.pad_token_id

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids or inputs_embeds must not be None")
        seq_length = input_shape[1]

        if position_ids is None:
            if self.offset_pos_ids:
                position_ids = self.create_position_ids_from_input_ids(input_ids)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
