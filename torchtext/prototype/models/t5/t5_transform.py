from typing import Any, List

import torch
import torch.nn as nn
import torchtext.transforms as T
from torchtext.data.functional import load_sp_model
from torchtext.functional import to_tensor
from torchtext.utils import get_asset_local_path


class T5Transform(nn.Module):
    def __init__(self, sp_model_path: str, max_seq_len: int, eos_idx: int, padding_idx: int):
        super().__init__()

        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))
        self.max_seq_len = max_seq_len
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        self.pipeline = T.Sequential(T.Truncate(self.max_seq_len), T.AddToken(token=self.eos_idx, begin=False))

    def forward(self, input: Any) -> Any:
        tokens = self.encode(input)
        out = to_tensor(self.pipeline(tokens), padding_value=self.padding_idx)
        return out

    def encode(self, input: Any) -> Any:
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
    def decode(self, input: Any) -> Any:
        if torch.jit.isinstance(input, List[List[int]]):
            tokens: List[str] = []
            for ids in input:
                tokens.append(self.sp_model.DecodeIds(ids))
            return tokens
        elif torch.jit.isinstance(input, List[int]):
            return self.sp_model.DecodeIds(input)
        else:
            raise TypeError("Input type not supported")
