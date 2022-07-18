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

import torch
import torch.nn as nn


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
