import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Union

__all__ = [
    'to_tensor',
    'truncate',
    'add_token',
]


def to_tensor(input: Union[List[int], List[List[int]]], padding_value: Optional[int] = None) -> Tensor:

    if isinstance(input[0], list):
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
    else:
        return torch.tensor(input, dtype=torch.long)


def truncate(input: Union[List[str], List[List[int]]], max_seq_len: int) -> Union[List[int], List[List[int]]]:
    if isinstance(input[0], list):
        output: List[List[int]] = []

        for ids in input:
            output.append(ids[:max_seq_len])

        return output
    else:
        return input[:max_seq_len]


def add_token(input: Union[List[int], List[List[int]]], token_id: int, begin: bool = True) -> Union[List[int], List[List[int]]]:
    if isinstance(input, list):
        output: List[List[int]] = []

        if begin:
            for ids in input:
                output.append([token_id] + ids)
        else:
            for ids in input:
                output.append(ids + [token_id])

        return output

    else:
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
