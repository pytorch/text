import logging
import math
from typing import Optional, List, Union

import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class PositionalEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, pad_index: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, pad_index)
        self.pad_index = pad_index

    def forward(self, input):
        positions = self._make_positions(input, self.pad_index)
        return self.embedding(positions)

    def max_positions(self):
        if self.pad_index is not None:
            return self.num_embeddings - self.pad_index - 1
        else:
            return self.num_embeddings

    def _make_positions(self, tensor, pad_index: int):
        masked = tensor.ne(pad_index).long()
        return torch.cumsum(masked, dim=1) * masked + pad_index


class ResidualMLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation=nn.GELU,
        add_residual=True,
    ):
        super().__init__()
        modules = []
        for last_dim, dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.extend(
                [nn.Linear(last_dim, dim), activation(), nn.Dropout(dropout)]
            )

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        modules.extend([nn.Linear(last_dim, input_dim), nn.Dropout(dropout)])

        self.mlp = nn.Sequential(*modules)
        self.add_residual = add_residual

    def forward(self, input):
        bias = self.mlp(input)
        if not hasattr(self, "add_residual"):
            self.add_residual = True
        if self.add_residual:
            return input + bias
        else:
            return bias


class MultiheadSelfAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        scaling: Optional[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        expected_scaling = float(1 / math.sqrt(self.head_dim))

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim={embed_dim} should be a multiple of num_heads={num_heads}"

        if not scaling:
            logger.warn(
                f"""
                Scaling not set. Please manually set scaling for transformers.
                In this case the suggested value {expected_scaling} will be inferred,
                or float(1 / math.sqrt(head_dim))
                where head_dim = embed_dim // num_heads = {self.head_dim}
                and embed_dim = {embed_dim} and num_heads = {num_heads}.
                """
            )
            scaling = expected_scaling

        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        target_length, batch_size, embed_dim = query.size()
        mask_batch_size, source_length = key_padding_mask.size()

        torch._assert(embed_dim == self.embed_dim, "query embed dim doesn't match")
        torch._assert(
            batch_size == mask_batch_size,
            "query and key_padding_mask batch sizes differed",
        )

        projection = self.input_projection(query)
        q, k, v = projection.chunk(3, dim=-1)
        q = self.scaling * q

        batch_heads = batch_size * self.num_heads

        q = q.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)

        torch._assert(
            k.size(1) == source_length, "key size should be equal to source length"
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            torch._assert(attn_mask.dim() == 2, "Expected attn_mask of dim 2 but got {}".format(attn_mask.dim()))
            torch._assert(attn_mask.size(0) == target_length, "attn_mask shape didn't match for target length {}".format(target_length))
            torch._assert(attn_mask.size(1) == source_length, "attn_mask shape didn't match for source length {}".format(source_length))
            torch._assert(attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f"Only float or bool types are supported for attn_mask not {attn_mask.dtype}")
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype)
                new_attn_mask.masked_fill_(attn_mask, -1e8 if query.dtype == torch.float32 else -1e4)
                attn_mask = new_attn_mask
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        torch._assert(attn_weights.dim() == 3, "Unexpected attn_weights dim")
        torch._assert(
            attn_weights.size(0) == batch_heads,
            "attn_weights shape didn't match for batch heads",
        )
        torch._assert(
            attn_weights.size(1) == target_length,
            "attn_weights shape didn't match for target length",
        )
        torch._assert(
            attn_weights.size(2) == source_length,
            "attn_weights shape didn't match for source length",
        )

        attn_weights = attn_weights.view(
            batch_size, self.num_heads, target_length, source_length
        )
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attn_weights = attn_weights.view(batch_heads, target_length, source_length)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)

        torch._assert(
            attn.dim() == 3,
            "unexpected attn dim size",
        )
        torch._assert(
            attn.size(0) == batch_heads,
            "attn shape didn't match for batch heads",
        )
        torch._assert(
            attn.size(1) == target_length,
            "attn shape didn't match for target length",
        )
        torch._assert(
            attn.size(2) == self.head_dim,
            "attn shape didn't match for head dim",
        )
        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(target_length, batch_size, self.head_dim * self.num_heads)
        )
        attn = self.output_projection(attn)

        return attn


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        ffn_dimension: Optional[int] = None,
        dropout: float = 0.1,
        normalize_before: bool = False,
        scaling: Optional[float] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiheadSelfAttention(
            embedding_dim,
            num_heads=num_attention_heads,
            scaling=scaling,
            dropout=dropout,
        )

        self.residual_mlp = ResidualMLP(
            embedding_dim,
            hidden_dims=[ffn_dimension or embedding_dim * 4],
            add_residual=not normalize_before,
        )

        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.normalize_before = normalize_before

    def forward(self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            torch._assert(attn_mask.dim() == 2, "Expected attn_mask of dim 2 but got {}".format(attn_mask.dim()))
            torch._assert(attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f"Only float or bool types are supported for attn_mask not {attn_mask.dtype}")

        if not hasattr(self, "normalize_before"):
            self.normalize_before = False

        if self.normalize_before:
            x = self.attention_layer_norm(input)
            attention = self.attention(x, key_padding_mask, attn_mask)
            attention = self.dropout(attention)
            biased_input = input + attention
            x = self.final_layer_norm(biased_input)
            return self.residual_mlp(x) + biased_input
        else:
            attention = self.attention(input, key_padding_mask, attn_mask)
            attention = self.dropout(attention)
            biased_input = input + attention
            biased_input = self.attention_layer_norm(biased_input)
            biased = self.residual_mlp(biased_input)
            return self.final_layer_norm(biased)


class TransformerEncoder(Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        max_seq_len: int,
        num_encoder_layers: int,
        num_attention_heads: int,
        ffn_dimension: Optional[int] = None,
        dropout: float = 0.1,
        normalize_before: bool = False,
        scaling: Optional[float] = None,
        return_all_layers: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    num_attention_heads=num_attention_heads,
                    ffn_dimension=ffn_dimension,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    scaling=scaling,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.positional_embedding = PositionalEmbedding(
            max_seq_len, embedding_dim, padding_idx
        )
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.return_all_layers = return_all_layers

    def forward(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        if attn_mask is not None:
            torch._assert(attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f"Only float or bool types are supported for attn_mask not {attn_mask.dtype}")

        padding_mask = tokens.eq(self.padding_idx)

        token_embeddings = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)

        embedded = token_embeddings + embedded_positions

        if not hasattr(self, "normalize_before"):
            self.normalize_before = False
        if not self.normalize_before:
            embedded = self.embedding_layer_norm(embedded)
        embedded = self.dropout(embedded)

        padded_embedded = embedded * (1 - padding_mask.unsqueeze(-1).type_as(embedded))

        encoded = padded_embedded.transpose(0, 1)

        if self.return_all_layers:
            states = [encoded]

            for layer in self.layers:
                encoded = layer(encoded, padding_mask, attn_mask)
                states.append(encoded)

            if self.normalize_before:
                for i, state in enumerate(states):
                    states[i] = self.embedding_layer_norm(state)

            # states are returned as T x B x C
            return states
        else:
            for layer in self.layers:
                encoded = layer(encoded, padding_mask, attn_mask)

            if self.normalize_before:
                encoded = self.embedding_layer_norm(encoded)

            # states are returned as T x B x C
            return encoded
