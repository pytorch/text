# /* Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. */
from typing import List, Union

import torch
import torch.nn as nn
import torchtext.transforms as T
from torchtext.data.functional import load_sp_model
from torchtext.functional import to_tensor
from torchtext.utils import get_asset_local_path


class T5Transform(nn.Module):
    """
    This transform makes use of a pre-trained sentencepiece model to tokenize text input. The resulting output is fed to the T5 model.

    Additional details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str
    :param max_seq_len: Maximum sequence length accepted for inputs to T5 model
    :type max_seq_len: int
    :param eos_idx: End-of-sequence token id
    :type eos_idx: int
    :param padding_idx: Padding token id
    :type padding_idx: int

    Example
        >>> from torchtext.prototype.models import T5Transform
        >>> transform = T5Transform("spm_model", max_seq_len = 10, eos_idx = 1, padding_idx = 0)
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str, max_seq_len: int, eos_idx: int, padding_idx: int):
        super().__init__()
        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))
        self.max_seq_len = max_seq_len
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        self.pipeline = T.Sequential(T.Truncate(self.max_seq_len - 1), T.AddToken(token=self.eos_idx, begin=False))

    def forward(self, input: Union[str, List[str]]) -> torch.Tensor:
        """
        :param input: Input sentence or list of sentences to tokenize.
        :type input: Union[str, List[str]]
        :return: Tokenized text that has been truncated, appended with end-of-sequence token, and padded
        :rtype: torch.Tensor
        """
        tokens = self.encode(input)
        out = to_tensor(self.pipeline(tokens), padding_value=self.padding_idx)
        return out

    @torch.jit.export
    def encode(self, input: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        :param input: Input sentence or list of sentences to tokenize.
        :type input: Union[str, List[str]]
        :return: Tokenized text that has been translated to token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[int]] = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsIds(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsIds(input)
        else:
            raise TypeError("Input type not supported")

    @torch.jit.export
    def decode(self, input: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        :param input: List of token ids or list of lists of token ids (i.e. batched).
        :type input: Union[List[int], List[List[int]]]
        :return: Sentence or list of sentencess that were translated from the input token ids
        :rtype: Union[str, List[str]]
        """
        if torch.jit.isinstance(input, List[List[int]]):
            tokens: List[str] = []
            for ids in input:
                tokens.append(self.sp_model.DecodeIds(ids))
            return tokens
        elif torch.jit.isinstance(input, List[int]):
            return self.sp_model.DecodeIds(input)
        else:
            raise TypeError("Input type not supported")
