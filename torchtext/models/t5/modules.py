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

# Parts of code are originally from the HuggingFace team and can be found here
# https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py
# */

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

# Define reusable types for past_key_values
PAST_KEY_VALUES_TYPE = Tuple[Tensor, Tensor, Tensor, Tensor]
PAST_KEY_VALUE_TYPE = Tuple[Tensor, Tensor]
# If running forward pass in encoder only, there won't be KVs from cross-attention therefore we need a version with optional tensors
PAST_KEY_VALUES_UNFILLED_TYPE = Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]

# Define reusable types for encoder/decoder outputs
SEQ_2_SEQ_OUTPUTS_TYPE = Dict[
    str, Union[Optional[Tensor], List[Tensor], List[Optional[Tensor]], Optional[List[PAST_KEY_VALUES_UNFILLED_TYPE]]]
]


class T5MultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_decoder: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        qkv_dim: int = 64,
        compute_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        r"""T5MultiheadAttention based on `nn.MultiheadAttention`.

        Args:
            embed_dim: Total dimension of the model.
            num_heads: Parallel attention heads.
            is_decoder: Whether or not multihead attention is being performed on a decoder layer. Default: `False`
            dropout: Probability of an element to be zeroed. Default: 0.0
            bias: If specified, adds bias to input / output projection layers. Default: `False`.
            qkv_dim: Projection dimension (per head) for query, keys, and values. Defualt: 64.
            compute_relative_attention_bias: Whether or not the relative position embeddings
                need to be computed. Wypically occurs in the first layer of the encoder/decoder
                and the resulting position embeddings are returned to be passed up to higher layers. (defualt: False)
            relative_attention_num_buckets: Number of relative position buckets. Default: `32`
            relative_attention_max_distance: Maximum threshold on the relative distance used to
                allocate buckets. Anything larger gets placed in the same bucket. Default: `128`
        """
        super().__init__(embed_dim, num_heads, dropout, bias, False, False, qkv_dim, qkv_dim, True, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.is_decoder = is_decoder
        self.inner_dim = qkv_dim * num_heads
        self.q_proj_weight = nn.Parameter(torch.empty((self.inner_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((self.inner_dim, embed_dim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((self.inner_dim, embed_dim), **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(self.inner_dim, embed_dim, bias=bias, **factory_kwargs)

        self.register_parameter("in_proj_weight", None)

        self.compute_relative_attention_bias = compute_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        if compute_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        else:
            self.relative_attention_bias = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_length: Optional[int] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = False,
        position_bias: Optional[Tensor] = None,
        past_key_value: Optional[PAST_KEY_VALUE_TYPE] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        r"""Allows the model to jointly attend to information from different representation subspaces
        as described in the paper: `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`.
        Also incorporates relative attention bias when computing attention scores as descripted in the paper:
        `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer <https://arxiv.org/pdf/1910.10683.pdf>`.

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
            position_bias: Position bias tensor used if to add relative attention bias to attention scores. Default: `None`

        Returns:
            attn_output: Attention outputs of shape :math:`(N, L, E)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`E` is the embedding dimension `embed_dim`.
            attn_output_weights: Only returned when `need_weights=True`. If `average_attn_weights=True`,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If `average_weights=False`, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.
            position_bias: Used in attention scoring. Only computed when `compute_relative_attention_bias=True`
                and `position_bias=None`. Has shape :math:`(1, num_heads, L, S)`.
            key_value: Calculated weights for keys and values. Used for incremental decoding.
        """
        attn_output, position_bias, attn_output_weights, key_value = self._t5_multi_head_attention_forward(
            query,
            key,
            value,
            query_length=query_length,
            position_bias=position_bias,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            past_key_value=past_key_value,
        )
        return attn_output, position_bias, attn_output_weights, key_value

    def _t5_multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_length: Optional[int],
        position_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = False,
        past_key_value: Optional[PAST_KEY_VALUE_TYPE] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4909."""
        is_self_attention = torch.equal(query, key)
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
        real_seq_length = tgt_len

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        src_len = real_seq_length if is_self_attention else key.shape[1]

        assert (
            embed_dim == self.embed_dim
        ), f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        head_dim = self.inner_dim // self.num_heads
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

        q, k, v = self._t5_in_projection(
            query,
            key,
            value,
            bsz,
            head_dim,
            self.q_proj_weight,
            self.k_proj_weight,
            self.v_proj_weight,
            b_q,
            b_k,
            b_v,
            is_self_attention,
            past_key_value,
        )

        if attn_mask is None:
            if self.is_decoder:
                if is_self_attention:
                    attn_mask = torch.triu(
                        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=query.device), diagonal=1
                    )
                else:
                    attn_mask = torch.zeros((tgt_len, src_len), device=query.device)
            else:
                attn_mask = torch.zeros((src_len, src_len), device=query.device, dtype=torch.bool)

        # Prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask is not supported. Using bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert (
                    attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
                ), f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
            if attn_mask.dim() == 2:
                x, y = attn_mask.shape
                attn_mask = attn_mask.view(1, 1, x, y).expand(bsz, self.num_heads, -1, -1)
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # Prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask is not supported. Using bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # Reshape q, k, v for multihead attention and make them batch first
        q = q.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        if past_key_value is None:
            k = k.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)

        # Have to check this after resize
        src_len = k.size(2)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, src_len):
                # It's possible that padding mask only takes into acct curr tgt_length instead of past_key_value
                assert (
                    past_key_value is not None
                ), "Must provide past_key_value if key_padding_mask needs to be expanded."
                key_padding_mask = key_padding_mask.expand(bsz, src_len)

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
            tmp_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            attn_mask = tmp_attn_mask.masked_fill(attn_mask, float("-inf"))

        # Adjust dropout probability
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout

        # Modification to torch.nn.functional._multi_head_attention_forward to incorporate relative attention bias
        if position_bias is None:
            if not self.compute_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.num_heads, real_seq_length, src_len), device=k.device, dtype=k.dtype
                )
            else:
                position_bias = self._compute_bias(
                    real_seq_length, src_len, bidirectional=(not self.is_decoder), device=k.device
                )

            if past_key_value is not None:
                position_bias = position_bias[:, :, -query.size(1) :, :]

        # Always return KV pair; let user discard if they don't want it
        new_key_val = (k, v)

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

            return attn_output, position_bias, attn_output_weights, new_key_val

        else:
            if not is_batched:
                # Squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)

            return attn_output, position_bias, None, new_key_val

    def _t5_in_projection(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bsz: int,
        head_dim: int,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        b_q: Optional[Tensor] = None,
        b_k: Optional[Tensor] = None,
        b_v: Optional[Tensor] = None,
        is_self_attention: bool = True,
        past_key_value: Optional[PAST_KEY_VALUE_TYPE] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Performs the in-projection step of the attention operation. This is simply
        a triple of linear projections, with shape constraints on the weights which
        ensure embedding dimension uniformity in the projected outputs.
        Output is a triple containing projection tensors for query, key and value.

        Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4761.

        Args:
            q, k, v: query, key and value tensors to be projected.
            w_q, w_k, w_v: weights for q, k and v, respectively.
            b_q, b_k, b_v: optional biases for q, k and v, respectively.

        Shape:
            Inputs:
            - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
                number of leading dimensions.
            - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
                number of leading dimensions.
            - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
                number of leading dimensions.
            - w_q: :math:`(Ei, Eq)` where Ei is the dimension to which the query, key, and value
                emebeddings are to be projected
            - w_k: :math:`(Ei, Ek)`
            - w_v: :math:`(Ei, Ev)`
            - b_q: :math:`(Ei)`
            - b_k: :math:`(Ei)`
            - b_v: :math:`(Ei)`
            Output: in output triple :math:`(q', k', v')`,
            - q': :math:`[Qdims..., Ei]`
            - k': :math:`[Kdims..., Ei]`
            - v': :math:`[Vdims..., Ei]`
        """
        Eq, Ek, Ev = query.size(-1), key.size(-1), value.size(-1)
        assert w_q.shape == (
            self.inner_dim,
            Eq,
        ), f"expecting query weights shape of {(self.inner_dim, Eq)}, but got {w_q.shape}"
        assert w_k.shape == (
            self.inner_dim,
            Ek,
        ), f"expecting key weights shape of {(self.inner_dim, Ek)}, but got {w_k.shape}"
        assert w_v.shape == (
            self.inner_dim,
            Ev,
        ), f"expecting value weights shape of {(self.inner_dim, Ev)}, but got {w_v.shape}"
        assert b_q is None or b_q.shape == (
            self.inner_dim,
        ), f"expecting query bias shape of {(self.inner_dim,)}, but got {b_q.shape}"
        assert b_k is None or b_k.shape == (
            self.inner_dim,
        ), f"expecting key bias shape of {(self.inner_dim,)}, but got {b_k.shape}"
        assert b_v is None or b_v.shape == (
            self.inner_dim,
        ), f"expecting value bias shape of {(self.inner_dim,)}, but got {b_v.shape}"
        query_proj = F.linear(query, w_q, b_q)

        if is_self_attention:
            # Self-attention over query (hidden states)
            key_proj = F.linear(query, w_k, b_k)
            value_proj = F.linear(query, w_v, b_v)
        else:
            if past_key_value is None:
                # Cross-attention (over current key/val states)
                key_proj = F.linear(key, w_k, b_k)
                value_proj = F.linear(value, w_v, b_v)
            else:
                # Should never reach this branch
                key_proj = key
                value_proj = value

        if past_key_value is not None:
            if is_self_attention:
                # Concat old key vals w/ new calculated ones for speed in decoding
                key_proj = key_proj.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
                value_proj = value_proj.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
                key_proj = torch.cat([past_key_value[0], key_proj], dim=2)
                value_proj = torch.cat([past_key_value[1], value_proj], dim=2)
            else:
                # Cross-attention context
                key_proj = past_key_value[0]
                value_proj = past_key_value[1]

        assert key_proj is not None
        assert value_proj is not None

        return query_proj, key_proj, value_proj

    def _t5_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        position_bias: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        r"""Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.

        Modified from https://github.com/pytorch/pytorch/blob/5953fd9133c0bdcc0158acf1472fac403bc5f636/torch/nn/functional.py#L4814.

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

        Returns:
            Tensor pair containing attended values and attention weights.
        """
        B, H, _, E = q.shape
        # HF implementation does not perform this normalization. For the sake of matching test results, we have commented it out
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

    def _compute_bias(
        self, query_length: int, key_length: int, bidirectional: bool = True, device: Optional[torch.device] = None
    ) -> Tensor:
        """Compute binned relative position bias.

        Modified from https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L421.
        """
        assert self.relative_attention_bias is not None
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def _relative_position_bucket(
        self, relative_position: Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ) -> Tensor:
        r"""Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        and https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L374.

        Translates relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on.

        Args:
            relative_position: Tensor w/ initially constructed relative positions.
            bidirectional: If attention is bidirectional; when in decoder, this should be False.
            num_buckets: Number of buckets to utilize.
            max_distance: Maximum distance between positions.

        Returns:
            Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets).
        """
        relative_buckets = torch.zeros(relative_position.shape, dtype=torch.long, device=relative_position.device)
        if bidirectional:
            num_buckets = num_buckets // 2
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


class T5LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        r"""Construct a layernorm module in the T5 style. No bias and no subtraction of mean.

        Based on https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L239.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        r"""T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        half-precision inputs is done in fp32.

        Args:
            hidden_states: Tensor to be normalized. Final dimension must be model dimension (i.e. number of expected features in the input).

        Returns:
            Tensor with the same shape as hidden_states after having been normalized.
        """

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5Layer(nn.Module):
    r"""T5Layer is made up of a self-attn block, optional cross-attn block, and feed-forward network.

    This T5 layer is based on the paper:
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
    Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Journal of Machine Learning Research.
    Volume 21 Issue 140 pages 1-67. http://jmlr.org/papers/v21/20-074.html
    Users may modify or implement in a different way during application.

    Args:
        d_model: Number of expected features in the input (required).
        nhead: Number of heads in the multihead attention models (required).
        dim_feedforward: Dimension of the feedforward network model (default: 3072).
        qkv_dim: Projection dimension (per head) for query, keys, and values. (default: 64).
        dropout: Dropout value (default: 0.1).
        activation: Activation function of the intermediate layer, can be a string
            ("relu", "gelu", or "gelu_new") or a unary callable. (default: F.relu)
        is_gated_act: Option to include gated activated as done in FLAN-T5, see
            https://huggingface.co/google/flan-t5-xxl. (default: False)
        layer_norm_eps: The eps value in layer normalization components. (default=1e-6)
        relative_attention_num_buckets: Number of relative position buckets. (default: 32)
        relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket. (default: 128)
        compute_relative_attention_bias: Whether or not the relative position embeddings
            need to be computed. Typically occurs in the first layer of the encoder
            and resulting position embeddings are returned to be passed up to higher layers. (default: False)
        is_decoder: Whether the T5Layer will be instantiated as a decoder layer or encoder layer. (default: False)
        device: Device to use any newly constructed Tensors. (optional)
        dtype: Datatype to use on any newly constructed Tensors. (optional)

    Examples::
        >>> single_encoder_layer = T5Layer(d_model=768, nhead=12)
        >>> src = torch.rand(32, 20, 768)
        >>> single_encoder_layer(src)

        >>> single_decoder_layer = T5Layer(d_model=768, nhead=12, is_decoder=True)
        >>> src = torch.rand(32, 20, 768)
        >>> tgt = torch.rand(32, 1, 768)
        >>> single_decoder_layer(tgt, src)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 3072,
        qkv_dim: int = 64,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        is_gated_act: bool = False,
        layer_norm_eps: float = 1e-6,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        compute_relative_attention_bias: bool = False,
        is_decoder: bool = False,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.is_gated_act = is_gated_act
        self.compute_relative_attention_bias = compute_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.is_decoder = is_decoder

        self.self_attn = T5MultiheadAttention(
            d_model,
            nhead,
            is_decoder=is_decoder,
            dropout=dropout,
            qkv_dim=qkv_dim,
            compute_relative_attention_bias=compute_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            device=device,
            dtype=dtype,
        )

        if self.is_decoder:
            self.cross_attn = T5MultiheadAttention(
                d_model,
                nhead,
                is_decoder=True,
                dropout=dropout,
                qkv_dim=qkv_dim,
                compute_relative_attention_bias=False,
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                device=device,
                dtype=dtype,
            )
            self.norm3 = T5LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout4 = nn.Dropout(dropout)
        else:
            self.cross_attn = None
            self.norm3 = None
            self.dropout4 = None

        if self.is_gated_act:
            self.linear1 = None
            self.linear1_0 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear1_1 = nn.Linear(d_model, dim_feedforward, bias=False)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear1_0 = None
            self.linear1_1 = None

        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.norm1 = T5LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = T5LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            assert activation in (
                "relu",
                "gelu",
                "gelu_new",
            ), f"Do not support '{activation}' activation. Use 'relu' or 'gelu' or 'gelu_new'"
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
            elif activation == "gelu_new":
                # The following should match the math of https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
                self.activation = nn.GELU(approximate="tanh")
        else:
            self.activation = activation

    def forward(
        self,
        seq: Tensor,
        memory: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        seq_key_padding_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        past_key_values: Optional[PAST_KEY_VALUES_TYPE] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], PAST_KEY_VALUES_UNFILLED_TYPE]:
        r"""Pass the inputs (and mask) through the encoder layer.

        Args:
            seq: Input sequence (required).
                Must have shape (B, Ns, E) where B is the batch size, Nt is the sequence length,
                and E is the model dimension. This will be the src sequence if `self.is_decoder = False`
                and tgt sequence if `self.is_decoder = True`.
            memory: Encoder sequence (optional).
                Output from encoder layer, only needs to be included when in decoding context.
            mask: Attention mask for self-attention. (optional).
                Must have shape (Ns, Ns).
            seq_key_padding_mask: Mask for the seq keys per batch (optional).
                Must have shape (B, Ns).
            memory_mask: Attention mask for attention in decoding context. (optional)
                Must have shape (Nm, Nm).
            memory_key_padding_mask: Mask for the memory keys per batch (optional).
                Must have shape (B, Ns).
            position_bias: Relative attention bias to be used when computing self-attention scores (optional)
                Must have shape (B, H, Ns, Ns) where H is the number of heads.
            past_key_values: Past key values used for incremental decoding (optional).
                Tuple with Tensors of shape (B, H, N)>>>>> Check this????

        Returns:
            Tuple of Tensors being hidden states, position bias, self-attention scores, cross-attention scores,
                and key-value pairs.
        """
        # See Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if past_key_values is not None:
            self_attn_past_key_value = past_key_values[:2]
            cross_attn_past_key_value = past_key_values[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        x = seq
        sa_out, position_bias, sa_scores, sa_kv = self._sa_block(
            self.norm1(x), mask, seq_key_padding_mask, position_bias, self_attn_past_key_value
        )
        x = x + sa_out

        if self.is_decoder:
            assert memory is not None, "Must provide memory (encoder hidden states)."
            assert self.norm3 is not None
            query_length = sa_kv[0].shape[2]
            ca_out, ca_scores, ca_kv = self._ca_block(
                self.norm3(x), memory, query_length, memory_mask, memory_key_padding_mask, cross_attn_past_key_value
            )
            x = x + ca_out
        else:
            ca_scores, ca_kv = None, None

        x = x + self._ff_block(self.norm2(x))

        new_key_value = sa_kv + (
            ca_kv
            if ca_kv is not None
            else (
                None,
                None,
            )
        )

        assert torch.jit.isinstance(new_key_value, PAST_KEY_VALUES_UNFILLED_TYPE)

        return x, position_bias, sa_scores, ca_scores, new_key_value

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        position_bias: Optional[Tensor],
        past_key_value: Optional[PAST_KEY_VALUE_TYPE] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Self-attention block."""
        attn, curr_position_bias, scores, curr_key_value = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            position_bias=position_bias,
            past_key_value=past_key_value,
        )

        if self.compute_relative_attention_bias:
            position_bias = curr_position_bias

        return self.dropout1(attn), position_bias, scores, curr_key_value

    def _ca_block(
        self,
        x: Tensor,
        mem: Tensor,
        query_length: Optional[int],
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        past_key_value: Optional[PAST_KEY_VALUE_TYPE] = None,
    ) -> Tuple[Tensor, Optional[Tensor], PAST_KEY_VALUE_TYPE]:
        """Cross-attention block."""
        assert self.cross_attn is not None
        assert self.dropout4 is not None
        attn, _, scores, curr_key_value = self.cross_attn(
            x,
            mem,  # Pass in memory (enc) states as keys
            mem,  # Pass in memory (enc) states as values
            query_length=query_length,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            past_key_value=past_key_value,
        )
        return self.dropout4(attn), scores, curr_key_value

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward block."""
        if self.is_gated_act:
            assert self.linear1_0 is not None
            assert self.linear1_1 is not None
            wi_0 = self.activation(self.linear1_0(x))
            wi_1 = self.linear1_1(x)
            hidden_states = wi_0 * wi_1
            hidden_states = self.dropout2(hidden_states)
            hidden_states = self.linear2(hidden_states)
        else:
            assert self.linear1 is not None
            hidden_states = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout3(hidden_states)


class T5Encoder(nn.Module):
    """T5Encoder is a stack of N encoder layers.

    Args:
        d_model: Number of expected features in the input (required).
        nhead: Number of heads in the multihead attention models (required).
        num_layers: Number of encoder layers in the stack (required)
        dim_feedforward: Dimension of the feedforward network model (default=3072).
        qkv_dim: Projection dimension (per head) for query, keys, and values. (defualt=64).
        dropout: Dropout value (default=0.1).
        activation: Activation function of the intermediate layer, can be a string
            ("relu", "gelu", or "gelu_new") or a unary callable. (default: F.relu)
        is_gated_act: Option to include gated activated as done in FLAN-T5, see
            https://huggingface.co/google/flan-t5-xxl. (default: False)
        layer_norm_eps: The eps value in layer normalization components (default=1e-6).
        relative_attention_num_buckets: Number of relative position buckets (default: 32)
        relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket (defulat: 128)
        token_embeddings (nn.Module): Embedding layer to be passed in the case that the input to `forward`
            is not already embedded.
        device: Device to use any newly constructed Tensors. (optional)
        dtype: Datatype to use on any newly constructed Tensors. (optional)

    Examples::
        >>> encoder = T5Encoder(d_model=768, nhead=12, num_layers=12)
        >>> tgt = torch.rand(32, 10, 512)
        >>> encoder(tgt)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 3072,
        qkv_dim: int = 64,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        is_gated_act: bool = False,
        layer_norm_eps: float = 1e-6,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        token_embeddings: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.token_embeddings = token_embeddings
        self.layers = nn.ModuleList(
            [
                T5Layer(
                    d_model,
                    nhead,
                    dim_feedforward=dim_feedforward,
                    qkv_dim=qkv_dim,
                    dropout=dropout,
                    activation=activation,
                    is_gated_act=is_gated_act,
                    layer_norm_eps=layer_norm_eps,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    relative_attention_max_distance=relative_attention_max_distance,
                    compute_relative_attention_bias=True if i == 0 else False,
                    is_decoder=False,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        embedded_src: Optional[Tensor] = None,
    ) -> SEQ_2_SEQ_OUTPUTS_TYPE:
        r"""Pass the input (and masks) through the stack of encoder layers.

        Args:
            src (Optional[Tensor]): Tokenized input sequence to the encoder.
                Must be batch first with shape (B, Ne) where B is the batch size and Ne is the
                encoder input sequence length.
            mask (Optional[Tensor]): Attention mask for self-attention.
                Must have shape (Nt, Nt).
            src_key_padding_mask (Optional[Tensor]): Mask for the tgt keys per batch.
                Must have shape (B, Nt).
            embedded_src (Optional[Tensor]): Embedded input sequence to the encoder layer.
                Must have shape (B, Nt, E) where B is the batch size, Nt is the target sequence
                length, and E is the model dimension.
                *Note*: If you do not provide this `embedded_tgt`, you must have provided a `token_embedding` layer \
                    in the initialization of the T5Encoder.

        Returns:
            Dictionary of last hidden layer, all hidden layers, position bias, and self-attention scores.
        """
        # This keeps the encoder self-contained and easy to use individually
        if embedded_src is None:
            assert (
                self.token_embeddings is not None and src is not None
            ), "Must provide `token_embeddings` and `tgt` if not providing already embedded tokens."
            embedded_src = self.token_embeddings(src)

        output = self.dropout1(embedded_src)
        position_bias = None
        all_outputs = torch.jit.annotate(List[Tensor], [])
        all_sa_scores = torch.jit.annotate(List[Optional[Tensor]], [])
        for mod in self.layers:
            all_outputs.append(output)
            output, position_bias, sa_score, _, _ = mod(
                output,
                mask=mask,
                seq_key_padding_mask=src_key_padding_mask,
                position_bias=position_bias,
            )
            all_sa_scores.append(sa_score)

        output = self.norm(output)
        output = self.dropout2(output)

        all_outputs.append(output)

        return {
            "encoder_output": output,
            "encoder_hidden_states": all_outputs,
            "encoder_position_bias": position_bias,
            "encoder_sa_scores": all_sa_scores,
        }


class T5Decoder(nn.Module):
    r"""T5Decoder is a stack of N decoder layers.

    Args:
        d_model: Number of expected features in the input (required).
        nhead: Number of heads in the multihead attention models (required).
        num_layers: Number of decoder layers in the stack (required)
        dim_feedforward: Dimension of the feedforward network model (default=3072).
        qkv_dim: Projection dimension (per head) for query, keys, and values. (defualt=64).
        dropout: Dropout value (default=0.1).
        activation: Activation function of the intermediate layer, can be a string
            ("relu", "gelu", or "gelu_new") or a unary callable. (default: F.relu)
        is_gated_act: Option to include gated activated as done in FLAN-T5, see
            https://huggingface.co/google/flan-t5-xxl. (default: False)
        layer_norm_eps: The eps value in layer normalization components (default=1e-6).
        relative_attention_num_buckets: Number of relative position buckets (default: 32)
        relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket (defulat: 128)
        device: Device to use any newly constructed Tensors. (optional)
        dtype: Datatype to use on any newly constructed Tensors. (optional)


    Examples::
        >>> decoder = T5Decoder(d_model=768, nhead=12, num_layers=12)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 1, 512)
        >>> decoder(tgt, memory)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 3072,
        qkv_dim: int = 64,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        is_gated_act: bool = False,
        layer_norm_eps: float = 1e-6,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                T5Layer(
                    d_model,
                    nhead,
                    dim_feedforward=dim_feedforward,
                    qkv_dim=qkv_dim,
                    dropout=dropout,
                    activation=activation,
                    is_gated_act=is_gated_act,
                    layer_norm_eps=layer_norm_eps,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    relative_attention_max_distance=relative_attention_max_distance,
                    compute_relative_attention_bias=True if i == 0 else False,
                    is_decoder=True,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(
        self,
        embedded_tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[PAST_KEY_VALUES_TYPE]] = None,
        return_past_key_values: bool = False,
    ) -> SEQ_2_SEQ_OUTPUTS_TYPE:
        r"""Pass the inputs (and masks) through the stack of decoder layers.

        Args:
            embedded_tgt: Input sequence to the decoder layer. (required).
                Must have shape (B, Nt, E) where B is the batch size, Nt is the target sequence
                length, and E is the model dimension.
            memory: Sequence from the last layer of the encoder. (required).
                Must have shape (B, Nts, E) where B is the batch size, Ns is the source sequence
                length, and E is the model dimension.
            tgt_mask: Attention mask for self-attention. (optional).
                Must have shape (Nt, Nt).
            memory_mask: Attention mask for cross-attention (optional).
                Must have shape (Nt, Ns).
            tgt_key_padding_mask: Mask for the tgt keys per batch (optional).
                Must have shape (B, Nt).
            memory_key_padding_mask: Mask for the memory keys per batch (optional).
                Must have shape (B, Ns).
            past_key_values: Past key values used for incremental decoding (optional).
                List of Tuple with Tensors of shape (B, H, N)>>>>> Check this????
            return_past_key_values: Boolean stating whether to return past_key_values from model. (default: False)

        Returns:
            Dictionary of last hidden state, all hidden states, position bias, self-attention scores, cross-attention scores
                and past key values (if requested).
        """
        output = self.dropout1(embedded_tgt)
        position_bias = None
        all_outputs = torch.jit.annotate(List[Tensor], [])
        all_sa_scores = torch.jit.annotate(List[Optional[Tensor]], [])
        all_ca_scores = torch.jit.annotate(List[Optional[Tensor]], [])
        all_key_values = torch.jit.annotate(List[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]], [])
        for i, mod in enumerate(self.layers):
            all_outputs.append(output)
            output, position_bias, sa_score, ca_score, past_key_value = mod(
                output,
                memory,
                mask=tgt_mask,
                memory_mask=memory_mask,
                seq_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                position_bias=position_bias,
                past_key_values=past_key_values[i] if past_key_values is not None else None,
            )
            all_sa_scores.append(sa_score)
            all_ca_scores.append(ca_score)
            # TODO: Can pass in enc-dec position_bias to avoid recalculating in cross-attn
            if past_key_value is not None and return_past_key_values:
                all_key_values.append(past_key_value)

        output = self.norm(output)
        output = self.dropout2(output)
        all_outputs.append(output)

        return {
            "decoder_output": output,
            "decoder_hidden_states": all_outputs,
            "decoder_position_bias": position_bias,
            "decoder_sa_scores": all_sa_scores,
            "decoder_ca_scores": all_ca_scores,
            "past_key_values": all_key_values,
        }
