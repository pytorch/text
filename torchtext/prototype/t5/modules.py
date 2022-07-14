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

    def forward():
        pass

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

    # NOTE: Taken from https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L374
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
