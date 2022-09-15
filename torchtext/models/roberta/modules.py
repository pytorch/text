import logging
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import Module

logger = logging.getLogger(__name__)


class PositionalEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, pad_index: int) -> None:
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


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        ffn_dimension: Optional[int] = None,
        dropout: float = 0.1,
        normalize_before: bool = False,
        scaling: Optional[float] = None,
    ) -> None:
        super().__init__()
        # TODO Manually setting scaling is not allowed
        ffn_dimension = ffn_dimension or embedding_dim * 4

        self.better_transformer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dimension,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=normalize_before,
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        better_to_old_names = {
            "better_transformer.self_attn.in_proj_weight": "attention.input_projection.weight",
            "better_transformer.self_attn.in_proj_bias": "attention.input_projection.bias",
            "better_transformer.self_attn.out_proj.weight": "attention.output_projection.weight",
            "better_transformer.self_attn.out_proj.bias": "attention.output_projection.bias",
            "better_transformer.linear1.weight": "residual_mlp.mlp.0.weight",
            "better_transformer.linear1.bias": "residual_mlp.mlp.0.bias",
            "better_transformer.linear2.weight": "residual_mlp.mlp.3.weight",
            "better_transformer.linear2.bias": "residual_mlp.mlp.3.bias",
            "better_transformer.norm1.weight": "attention_layer_norm.weight",
            "better_transformer.norm1.bias": "attention_layer_norm.bias",
            "better_transformer.norm2.weight": "final_layer_norm.weight",
            "better_transformer.norm2.bias": "final_layer_norm.bias",
        }
        for better, old in better_to_old_names.items():
            better_name = prefix + better
            old_name = prefix + old
            if old_name in state_dict:
                state_dict[better_name] = state_dict[old_name]
                state_dict.pop(old_name)
            elif better_name in state_dict:
                # Do nothing
                pass
            elif strict:
                missing_keys.append(better_name)

        super(TransformerEncoderLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # torch.nn.TransformerEncodeLayer's attn_mask and key_padding_mask's
        # order is reversed
        return self.better_transformer(input.transpose(0, 1), attn_mask, key_padding_mask).transpose(0, 1)


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
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        ffn_dimension = ffn_dimension or 4 * embedding_dim
        layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=normalize_before,
        )
        self.layers = torch.nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=True,
            mask_check=False,
        )
        self.positional_embedding = PositionalEmbedding(max_seq_len, embedding_dim, padding_idx)
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.return_all_layers = return_all_layers

    def forward(
        self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if attn_mask is not None:
            torch._assert(
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool,
                f"Only float or bool types are supported for attn_mask not {attn_mask.dtype}",
            )

        padding_mask = tokens.eq(self.padding_idx)

        token_embeddings = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)

        embedded = token_embeddings + embedded_positions

        if not hasattr(self, "normalize_before"):
            self.normalize_before = False
        if not self.normalize_before:
            embedded = self.embedding_layer_norm(embedded)
        embedded = self.dropout(embedded)

        if self.return_all_layers:
            encoded = embedded
            # B x T x C
            # Then transpose back to T x B x C
            states = [encoded.transpose(1, 0)]
            for layer in self.layers.layers:
                encoded = layer(encoded, src_key_padding_mask=padding_mask, src_mask=attn_mask)
                encoded_t = encoded.transpose(1, 0)
                states.append(encoded_t)
            if self.normalize_before:
                for i, state in enumerate(states):
                    states[i] = self.embedding_layer_norm(state)
            return states
        else:
            # B x T x C
            # Then transpose back to T x B x C
            encoded = self.layers(embedded, src_key_padding_mask=padding_mask).transpose(1, 0)
            if self.normalize_before:
                encoded = self.embedding_layer_norm(encoded)
            return encoded

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        better_to_old_names = {
            "self_attn.in_proj_weight": "attention.input_projection.weight",
            "self_attn.in_proj_bias": "attention.input_projection.bias",
            "self_attn.out_proj.weight": "attention.output_projection.weight",
            "self_attn.out_proj.bias": "attention.output_projection.bias",
            "linear1.weight": "residual_mlp.mlp.0.weight",
            "linear1.bias": "residual_mlp.mlp.0.bias",
            "linear2.weight": "residual_mlp.mlp.3.weight",
            "linear2.bias": "residual_mlp.mlp.3.bias",
            "norm1.weight": "attention_layer_norm.weight",
            "norm1.bias": "attention_layer_norm.bias",
            "norm2.weight": "final_layer_norm.weight",
            "norm2.bias": "final_layer_norm.bias",
        }
        for i in range(self.layers.num_layers):
            for better, old in better_to_old_names.items():
                better_name = prefix + "layers.layers.{}.".format(i) + better
                old_name = prefix + "layers.{}.".format(i) + old
                if old_name in state_dict:
                    state_dict[better_name] = state_dict[old_name]
                    state_dict.pop(old_name)
                elif better_name in state_dict:
                    # Do nothing
                    pass
                elif strict:
                    missing_keys.append(better_name)

        super(TransformerEncoder, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
