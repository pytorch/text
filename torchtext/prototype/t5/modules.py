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

# Parts of code are originally from
# https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py
# */

import math
import warnings
from typing import Optional, Tuple

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
        kdim=None,
        vdim=None,
        device=None,
        dtype=None,
    ) -> None:
        r"""
        Args:
            embed_dim: Total dimension of the model.
            num_heads: Parallel attention heads.
            is_decoder: Whether or not multihead attention is being performed on a decoder layer. Default: `False`
            dropout: Probability of an element to be zeroed. Default: 0.0
            bias: If specified, adds bias to input / output projection layers. Default: `False`.
            kdim: Total number of features for keys. Default: `None` (uses `kdim=embed_dim`).
            vdim: Total number of features for values. Default: `None` (uses `vdim=embed_dim`).
        """
        super().__init__(embed_dim, num_heads, dropout, bias, False, False, kdim, vdim, True, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.is_decoder = is_decoder
        self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter("in_proj_weight", None)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = False,
        compute_relative_attention_bias=False,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        relative_attention_bias: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""
        Allows the model to jointly attend to information from different representation subspaces
        as described in the paper:
        `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
        Also incorporates relative attention bias when computing attention scores as descripted in the paper:
        `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer <https://arxiv.org/pdf/1910.10683.pdf>`_.

        Args:
            query: Query embeddings of shape :math:`(N, L, E_q)`, where :math:`N` is the batch size, :math:`L` is the target sequence length,
                and :math:`E_q` is the query embedding dimension `embed_dim`.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(N, S, E_k)`, where :math:`N` is the batch size, :math:`S` is the source sequence length,
                and :math:`E_k` is the key embedding dimension `kdim`.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(N, S, E_v)`, where :math:`N` is the batch size, :math:`S` is the source
                sequence length, and :math:`E_v` is the value embedding dimension `vdim`.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within `key`
                to ignore for the purpose of attention (i.e. treat as "padding").
                Binary masks are supported. For a binary mask, a `True` value indicates that the corresponding `key`
                value will be ignored for the purpose of attention.
            need_weights: If specified, returns `attn_output_weights` in addition to `attn_outputs`.
                Default: `True`.
            attn_mask: If specified, a 2D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)`, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch. Binary, and float masks are supported.
                For a binary mask, a `True` value indicates that the corresponding position is not allowed to attend.
                For a float mask, the mask values will be added to the attention weight. Default: `None`
            average_attn_weights: If true, indicates that the returned `attn_weights` should be averaged across
                heads. Otherwise, `attn_weights` are provided separately per head. Note that this flag only has an
                effect when `need_weights=True`. Default: `False` (i.e. average weights across heads)
            compute_relative_attention_bias: Whether or not the relative position embeddings
                need to be computed. Wypically occurs in the first layer of the encoder/decoder
                and the resulting position embeddings are returned to be passed up to higher layers. (defualt: False)
            relative_attention_num_buckets: Number of relative position buckets. Default: `32`
            relative_attention_max_distance: Maximum threshold on the relative distance used to
                allocate buckets. Anything larger gets placed in the same bucket. Default: `128`
            relative_attention_bias: nn.Embeding object used to compute relative position embeddings. Default: `None`
            position_bias: Position bias tensor used if to add relative attention bias to attention scores. Default: `None`
        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(N, L, E)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`E` is the embedding dimension `embed_dim`.
            - **attn_output_weights** - Only returned when `need_weights=True`. If `average_attn_weights=True`,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If `average_weights=False`, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.
            - **position_bias** - Used in attention scoring. Only computed when `compute_relative_attention_bias=True`
                and `position_bias=None`. Has shape :math:`(1, num_heads, L, S)`.
        """
        attn_output, position_bias, attn_output_weights = self._t5_multi_head_attention_forward(
            query,
            key,
            value,
            compute_relative_attention_bias=compute_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            relative_attention_bias=relative_attention_bias,
            position_bias=position_bias,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        return attn_output, position_bias, attn_output_weights

    # NOTE: Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4909
    def _t5_multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        compute_relative_attention_bias: bool,
        relative_attention_num_buckets: Optional[int],
        relative_attention_max_distance: Optional[int],
        relative_attention_bias: Optional[Tensor],
        position_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, self.num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # Unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # Set up shape vars
        bsz, tgt_len, embed_dim = query.shape
        _, src_len, _ = key.shape

        assert (
            embed_dim == self.embed_dim
        ), f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        if isinstance(embed_dim, Tensor):
            # Embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // self.num_heads
        assert (
            head_dim * self.num_heads == embed_dim
        ), f"embed_dim {embed_dim} not divisible by num_heads {self.num_heads}"
        # Allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"

        # Compute in-projection
        assert self.q_proj_weight is not None, "q_proj_weight is None"
        assert self.k_proj_weight is not None, "k_proj_weight is None"
        assert self.v_proj_weight is not None, "v_proj_weight is None"
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q, k, v = F._in_projection(
            query, key, value, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, b_q, b_k, b_v
        )

        # Prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask is not supported. Using bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert (
                    attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
                ), f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # Ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.view(1, 1, tgt_len, tgt_len).expand(bsz, self.num_heads, -1, -1)
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # Prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask is not supported. Using bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # Reshape q, k, v for multihead attention and make em batch first
        q = q.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        src_len = k.size(2)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, tgt_len, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # Convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # Adjust dropout probability
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout

        # NOTE: Modification to torch.nn.functional._multi_head_attention_forward to incorporate relative attention bias
        if position_bias is None:
            if not compute_relative_attention_bias:
                position_bias = torch.zeros(
                    (self.num_heads, tgt_len, src_len), device=k.device, dtype=k.dtype
                ).unsqueeze(0)
            else:
                position_bias = self._compute_bias(
                    tgt_len,
                    src_len,
                    relative_attention_bias,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    relative_attention_max_distance=relative_attention_max_distance,
                    bidirectional=(not self.is_decoder),
                    device=k.device,
                )

        # Calculate attention and out projection
        attn_output, attn_output_weights = self._t5_dot_product_attention(q, k, v, position_bias, attn_mask, dropout_p)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # Optionally average attention weights over heads
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

            if not is_batched:
                # Squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)

            return attn_output, position_bias, attn_output_weights

        else:
            if not is_batched:
                # Squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)

            return attn_output, position_bias, None

    # NOTE: Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4814
    def _t5_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        position_bias: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: Query, key and value tensors. See Shape section for shape details.
            attn_mask: Optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: Dropout probability. If greater than 0.0, dropout is applied.
            position_bias: Position bias used to incorporate realtive attention bias in attention scors
        Shape:
            - q: :math:`(B, H, Nt, E)` where B is the batch size, H is the number of heads, Nt is the target sequence length,
                and E is the head dimension.
            - key: :math:`(B, H, Ns, E)` where B is the batch size, H is the number of heads, Ns is the source sequence length,
                and E is the head dimension.
            - value: :math:`(B, H, Ns, E)` where B is the batch size, H is the number of heads, Ns is the source sequence length,
                and E is the head dimension.
            - attn_mask: a 4D tensor of shape :math:`(B, H, Nt, Ns)`
            - position_bias: :math:`(1, H, Nt, Ns)`
            - Output: attention values have shape :math:`(B, Nt, H*E)`; attention weights
                have shape :math:`(B, H, Nt, Ns)`
        """
        B, H, _, E = q.shape
        # NOTE: HF implementation does not perform this normalization. For the sake of matching test results, we have commented it out
        # q = q / math.sqrt(E)

        attn = torch.matmul(q, k.transpose(3, 2))

        # NOTE: modification from torch.nn.functional._scaled_dot_product_attention to incorporate relative attention bias
        position_bias = position_bias.repeat(B, 1, 1, 1)
        if attn_mask is not None:
            position_bias += attn_mask
        attn += position_bias

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, -1, H * E)
        return output, attn

    # NOTE: modified from https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L421
    def _compute_bias(
        self,
        query_length: int,
        key_length: int,
        relative_attention_bias: Tensor,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        bidirectional: bool = True,
        device=None,
    ) -> Tensor:
        """Compute binned relative position bias"""
        if device is None:
            device = relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=bidirectional,
            num_buckets=relative_attention_num_buckets,
            max_distance=relative_attention_max_distance,
        )
        values = relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    # NOTE: Taken from https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L374
    def _relative_position_bucket(
        self, relative_position: Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ) -> Tensor:
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
        # Ensure relative_position is in the range [0, inf)

        # Half of the buckets are for exact increments in positions
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
