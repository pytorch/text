import itertools
import os
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_chars(input, pad_idx=1):
    # get info on length on each sentences
    batch_sizes = [len(sent) for sent in input]
    # flattening the array first and convert them to tensor
    tx = list(map(torch.tensor, itertools.chain.from_iterable(input)))
    # pad all the chars
    ptx = pad_sequence(tx, True, pad_idx)
    # split according to the original length
    sptx = ptx.split(batch_sizes)
    # finally, merge them back with padding
    final_padding = pad_sequence(sptx, True, pad_idx)

    return final_padding


def pad_words(input, pad_idx=1):
    txt = list(map(torch.tensor, input))
    return pad_sequence(txt, True, pad_idx)


def seed_everything(seed: Optional[int] = None) -> int:
    """Function that sets seed for pseudo-random number generators  in:
        pytorch, python.random and sets PYTHONHASHSEED environment variable.
        Imported from pytorch-lightning module
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
