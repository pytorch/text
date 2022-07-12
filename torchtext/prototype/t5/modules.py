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
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def t5_multi_head_attention_forward(
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
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, self.num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == self.embed_dim, f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
    if isinstance(embed_dim, Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(self.num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // self.num_heads
    assert head_dim * self.num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {self.num_heads}"
    # allow MHA to have different embedding dimensions when separate projection weights are used
    assert (
        key.shape[:2] == value.shape[:2]
    ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"

    #
    # compute in-projection
    #
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

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if self.bias_k is not None and self.bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert self.bias_k is None
        assert self.bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #

    q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * self.num_heads
        ), f"expecting static_k.size(0) of {bsz * self.num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.size(0) == bsz * self.num_heads
        ), f"expecting static_v.size(0) of {bsz * self.num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if self.add_zero_attn:
        zero_attn_shape = (bsz * self.num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(bsz * self.num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not self.training:
        dropout_p = 0.0
    else:
        dropout_p = self.dropout

    # NOTE: modification from torch.nn.functional._multi_head_attention_forward to incorporate relative attention bias
    if position_bias is None:
        if not compute_relative_attention_bias:
            position_bias = torch.zeros((self.num_heads, tgt_len, src_len), device=k.device, dtype=k.dtype).unsqueeze(0)
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

    # calculate attention and out projection
    attn_output, attn_output_weights = self._t5_scaled_dot_product_attention(
        q, k, v, position_bias, attn_mask, dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)

        return attn_output, attn_output_weights, position_bias

    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)

        return attn_output, None, position_bias


def _t5_scaled_dot_product_attention(
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
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        position_bias: position bias used to incorporate realtive attention bias in attention scors
    Shape:
        - q: :math:`(B, Nt, E)` where B is (batch size*num_heads), Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is (batch size*num_heads), Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is (batch size*num_heads), Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - position_bias: :math:`(1, num_heads, Nt, Ns)`
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    # NOTE: HF implementation does not perform this normalization. For the sake of matching test results, we have commented it out
    # q = q / math.sqrt(E)

    n_heads, tgt_len, src_len = position_bias.size()[1:]
    assert B % n_heads == 0
    assert tgt_len == Nt

    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    # NOTE: modification from torch.nn.functional._scaled_dot_product_attention to incorporate relative attention bias
    position_bias = position_bias.repeat(B // n_heads, 1, 1, 1)
    position_bias = position_bias.view(B, tgt_len, src_len)
    attn += position_bias

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


# NOTE: modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
def _compute_bias(
    query_length: int,
    key_length: int,
    relative_attention_bias: Tensor,
    relative_attention_num_buckets: int = 32,
    relative_attention_max_distance: int = 128,
    bidirectional: bool = True,
    device=None,
):
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


# NOTE: taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
def _relative_position_bucket(
    relative_position: Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
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
