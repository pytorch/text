import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

__all__ = [
    'to_tensor',
    'truncate',
    'add_token',
]


def to_tensor(input: List[List[int]], padding_value: Optional[int] = None) -> Tensor:
    if padding_value is None:
        output = torch.tensor(input, dtype=torch.long)
        return output
    else:
        output = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in input],
            batch_first=True,
            padding_value=float(padding_value)
        )
        return output


def truncate(input: List[List[int]], max_seq_len: int) -> List[List[int]]:
    output: List[List[int]] = []

    for ids in input:
        output.append(ids[:max_seq_len])

    return output


def add_token(input: List[List[int]], token_id: int, begin: bool = True) -> List[List[int]]:
    output: List[List[int]] = []

    if begin:
        for ids in input:
            output.append([token_id] + ids)
    else:
        for ids in input:
            output.append(ids + [token_id])

    return output
