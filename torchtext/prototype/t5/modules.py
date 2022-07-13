# /* Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original code is taken from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
# */

import math
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class T5MultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        is_decoder=False,
        dropout=0.0,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        r"""
        Args:
            embed_dim: total dimension of the model.
            num_heads: parallel attention heads.
            is_decoder: whether or not multihead attention is being performed on a decoder layer. Default: ``False``
            dropout: probability of an element to be zeroed. Default: 0.0
            bias: If specified, adds bias to input / output projection layers. Default: ``False``.
            add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
            add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
                Default: ``False``.
            kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
            vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        """
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.is_decoder = is_decoder
        self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter("in_proj_weight", None)

    def forward():
        pass

    # NOTE: taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    def _relative_position_bucket(
        self, relative_position: Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ):
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
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


# NOTE: Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
class T5LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5Layer(nn.Module):
    r"""T5Layer is made up of self-attn, cross-attn (decoder only) and feedforward network.
    This T5 layer is based on the paper:
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
    Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Journal of Machine Learning Research.
    Volume 21 Issue 140 pages 1-67. http://jmlr.org/papers/v21/20-074.html
    Users may modify or implement in a different way during application.
    Args:
        is_decoder: whether or not the layer belongs to the decoder. (required)
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. (default: relu)
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). (default: ``False``) (seq, batch, feature).
        relative_attention_num_buckets: the number of relative position buckets (default: 32)
        relative_attention_max_distance: maximum threshold on the relative distance used to
            allocate buckets. anything larger than that gets placed in the same bucket (default: 128)
        compute_relative_attention_bias: whether or not the relative position embeddings
            need to be computed. typically occurs in the first layer of encoder/decoder (default: False)
            and resulting position embeddings are returned to be passed up to higher layers.
        relative_attention_bias: tensor of weights to compute relative position embeddings. (default: None)

    Examples::
        >>> decoder_layer = T5Layer(is_decoder=True, d_model=768, nhead=12, batch_first=True)
        >>> memory = torch.rand(32, 10, 768)
        >>> tgt = torch.rand(32, 20, 768)
        >>> out = deoder_layer(tgt, memory)
    """

    def __init__(
        self,
        is_decoder: bool,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-6,
        batch_first: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        compute_relative_attention_bias: bool = False,
        relative_attention_bias: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.is_decoder = is_decoder
        self.compute_relative_attention_bias = compute_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = relative_attention_bias

        self.self_attn = T5MultiheadAttention(
            d_model, nhead, is_decoder=is_decoder, dropout=dropout, batch_first=batch_first, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.norm1 = T5LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = T5LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if is_decoder:
            self.cross_attn = T5MultiheadAttention(
                d_model,
                nhead,
                is_decoder=is_decoder,
                dropout=dropout,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
            self.norm3 = T5LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout4 = nn.Dropout(dropout)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
        else:
            self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the input sequence to the encoder/decoder layer (required).
            memory: the sequence from the last layer of the encoder (used for decoder only).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            position_bias: position embeddings to be used when computing attention scores (optional)
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = tgt
        sa_out, position_bias, sa_scores = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, position_bias)
        x = x + sa_out
        if self.is_decoder:
            ca_out, ca_scores = self._ca_block(self.norm3(x), memory, memory_mask, memory_key_padding_mask)
            x = x + ca_out
        x = x + self._ff_block(self.norm2(x))

        return x, position_bias, sa_scores, ca_scores if self.is_decoder else None

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        position_bias: Optional[Tensor],
    ) -> Tensor:
        attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            compute_relative_attention_bias=self.compute_relative_attention_bias,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance,
            relative_attention_bias=self.relative_attention_bias,
            position_bias=position_bias,
        )

        x = attn[0]
        scores = attn[1]
        if self.compute_relative_attention_bias and position_bias is None:
            position_bias = attn[2]

        return self.dropout1(x), position_bias, scores

    # cross attention block
    def _ca_block(
        self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        attn = self.cross_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        x = attn[0]
        scores = attn[1]
        return self.dropout4(x), scores

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout3(x)


class T5Stack(nn.Module):
    r"""T5 is a stack of N encoder/decoder layers
    Args:
        is_decoder: whether or not the layer belongs to the decoder. (required)
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        num_layers: the number of encoder/decoder layers in the stack (required)
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. (default: relu)
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). (default: ``False``) (seq, batch, feature).
        relative_attention_num_buckets: the number of relative position buckets (default: 32)
        relative_attention_max_distance: maximum threshold on the relative distance used to
            allocate buckets. anything larger than that gets placed in the same bucket (defulat: 128)
    Examples::
        >>> decoder = nn.T5Stack(is_decoder=True, d_model=768, nhead=12, num_layers=12)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 10, 512)
        >>> out = decoder(tgt, memory)
    """

    def __init__(
        self,
        is_decoder: bool,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-6,
        batch_first: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                T5Layer(
                    is_decoder,
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    layer_norm_eps,
                    batch_first,
                    relative_attention_num_buckets,
                    relative_attention_max_distance,
                    compute_relative_attention_bias=True if i == 0 else False,
                    relative_attention_bias=nn.Embedding(relative_attention_num_buckets, nhead) if i == 0 else None,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the input sequence to the encoder/decoder (required).
            memory: the sequence from the last layer of the encoder (for decoder only).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt
        position_bias = None
        all_outputs = ()
        sa_scores = ()
        ca_scores = ()
        for mod in self.layers:
            all_outputs = all_outputs + (output,)
            output, position_bias, sa_score, ca_score = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                position_bias=position_bias,
            )
            sa_scores = sa_scores + (sa_score,)
            ca_scores = ca_scores + (ca_score,)

        return output, all_outputs, position_bias, sa_scores, ca_scores
