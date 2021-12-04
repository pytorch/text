import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Union

__all__ = [
    'to_tensor',
    'truncate',
    'add_token',
]


def to_tensor(input: Union[List[int], List[List[int]]], padding_value: Optional[int] = None, dtype: Optional[torch.dtype] = torch.long) -> Tensor:
    if torch.jit.isinstance(input, List[int]):
        return torch.tensor(input, dtype=torch.long)
    else:
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(ids, dtype=dtype) for ids in input],
                batch_first=True,
                padding_value=float(padding_value)
            )
            return output


def truncate(input: Union[List[int], List[str], List[List[int]], List[List[str]]], max_seq_len: int) -> Union[List[int], List[str], List[List[int]], List[List[str]]]:
    if torch.jit.isinstance(input, List[int]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[str]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[str]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[List[int]]):
        output: List[List[int]] = []
        for ids in input:
            output.append(ids[:max_seq_len])
        return output
    else:
        output: List[List[str]] = []
        for ids in input:
            output.append(ids[:max_seq_len])
        return output


def add_token(input: Union[List[int], List[List[int]]], token_id: int, begin: bool = True) -> Union[List[int], List[List[int]]]:
    if torch.jit.isinstance(input, List[int]):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    else:
        output: List[List[int]] = []

        if begin:
            for ids in input:
                output.append([token_id] + ids)
        else:
            for ids in input:
                output.append(ids + [token_id])

        return output
