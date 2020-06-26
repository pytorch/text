import csv
import logging
import os

import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from torchtext.utils import (
    download_from_url,
    extract_archive
)

logger = logging.getLogger(__name__)


class Vocab(nn.Module):
    r"""Creates a vocab object which maps tokens to indices.

    Arguments:
        ordered_dict (collections.Counter): object holding the frequencies of each token found in the data.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        specials: The tuple of special tokens (e.g., padding or eos) that will be prepended to the vocabulary.
            The first value should always be the unknown token Default: ['<unk'>, '<pad>']
        specials_first: Whether to add special tokens into the vocabulary at first. If it is False,
            they are added into the vocabulary at last. Default: True.

    Raises:
        ValueError: if `vectors` is empty and a default `unk_tensor` isn't provided.
        RuntimeError: if `tokens` and `vectors` have different sizes or `tokens` has duplicates.
        TypeError: if all tensors within`vectors` are not of data type `torch.float`.
    """

    def __init__(self, counter, min_freq=1, specials=('<unk>', '<pad>'), specials_first=True):
        super(Vocab, self).__init__()

        if specials is None:
            raise ValueError("The specials list is empty and a default unk token wasn't provided.")

        # self.vectors = torch.classes.torchtext.Vocab(tokens, vectors, unk_tensor)

    @torch.jit.export
    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
        Returns:
            idx (Tensor): the index corresponding to the associated token.
        """
        return self.vectors.GetItem(token)

    @torch.jit.export
    def __setitem__(self, token: str, idx: int):
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
            vector (Tensor): a 1d tensor representing a vector associated with the token.

        Raises:
            TypeError: if `vector` is not of data type `torch.float`.
        """
        if vector.dtype != torch.float:
            raise TypeError("`vector` should be of data type `torch.float` but it's of type " + vector.dtype)

        self.vectors.AddItem(token, vector.float())
