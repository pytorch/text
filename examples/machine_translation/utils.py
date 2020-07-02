import itertools

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


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    char_tgt_batch, word_tgt_batch = zip(*tgt_batch)
    padded_src_batch = pad_chars(src_batch)
    padded_tgt_char_batch = pad_chars(char_tgt_batch)
    padded_tgt_word_batch = pad_words(word_tgt_batch)
    return (padded_src_batch, (padded_tgt_char_batch, padded_tgt_word_batch))
