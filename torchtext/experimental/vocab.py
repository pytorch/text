from collections import OrderedDict
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _infer_shape(f):
    num_lines = 0
    for line in f:
        num_lines += 1
    f.seek(0)
    return num_lines


def vocab_from_file_object(file_like_object, **kwargs):
    r"""Create a `Vocab` object from a file like object.

    Note that the tensor corresponding to each vector is of type `torch.float`.
    The `file_like_object` should contain tokens seperated by new lines in a csv file format.

    Format for csv file:
        token1
        token2
        ...
        token_n

    Args:
        file_like_object (FileObject): a file like object to read data from.
        Remaining keyword arguments: Passed to the constructor of Vocab class.

    Returns:
        Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.experimental.vocab import vocab_from_file_object
        >>> f = open('vocab.csv', 'r')
        >>> v = vocab_from_file_object(f, specials=('<unk>', '<pad>', '<eos>'), specials_first=False)
    """
    ordered_dict = OrderedDict()
    num_lines = _infer_shape(file_like_object)
    for line in tqdm(file_like_object, unit_scale=0, unit="lines", total=num_lines):
        token = line.rstrip()
        if token in ordered_dict:
            ordered_dict[token] += 1
        else:
            ordered_dict[token] = 1

    return Vocab(ordered_dict, **kwargs)


class Vocab(nn.Module):
    r"""Creates a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Arguments:
        ordered_dict (collections.OrderedDict): object holding the frequencies of each token found in the data.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        specials: The tuple of special tokens (e.g., padding or eos) that will be prepended/postpended to the vocabulary.
            based on the `specials_first` flag. The ordering of the tuple will be preserved. Default: ('<unk>', '<pad>')
        specials_first: Whether to add special tokens into the vocabulary at first. If it is False,
            they are added into the vocabulary at last. Default: True.

    Raises:
        ValueError: if a default `unk_token` isn't provided.

    Examples:
        >>> from torchtext.experimental.vocab import Vocab
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v = Vocab(ordered_dict)
    """

    def __init__(self, ordered_dict, min_freq=1, unk_token='<unk>', specials=('<unk>', '<pad>'), specials_first=True):
        super(Vocab, self).__init__()

        if not unk_token:
            raise ValueError("A default unk token wasn't provided.")

        if unk_token not in specials:
            raise ValueError("The unk token wasn't found in the `specials` tuple.")

        tokens = []
        for token, freq in ordered_dict.items():
            if freq >= min_freq:
                if token in specials:
                    raise ValueError("A `specials` token {} was found inside of `ordered_dict`."
                                     "Please ensure that the `ordered_dict` doesn't contain any special tokens.".format(token))
                tokens.append(token)

        # assume special tokens dont appear in ordered_dict
        if specials_first:
            tokens = list(specials) + tokens
        else:
            tokens += list(specials)

        self.vocab = torch.classes.torchtext.Vocab(tokens, unk_token)

    @torch.jit.export
    def __len__(self) -> int:
        r"""Returns:
            length (int): the length of the vocab
        """
        return len(self.vocab)

    @torch.jit.export
    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.

        Returns:
            index (int): the index corresponding to the associated token.
        """
        return self.vocab[token]

    @torch.jit.export
    def insert_token(self, token: str, index: int) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
            index (int): the index corresponding to the associated token.

        Raises:
            RuntimeError: if `index` not between [0, Vocab.size()] or if token already exists in the vocab.
        """
        self.vocab.insert_token(token, index)

    @torch.jit.export
    def append_token(self, token: str) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
        """
        self.vocab.append_token(token)

    @torch.jit.export
    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index (int): the index corresponding to the associated token.

        Returns:
            token (str): the token used to lookup the corresponding index.

        Raises:
            RuntimeError: if `index` not between [0, itos.size()].
        """
        return self.vocab.lookup_token(index)

    @torch.jit.export
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices (List[int]): the `indices` used to lookup their corresponding`tokens`.

        Returns:
            tokens (List[str]): the `tokens` associated with `indices`.

        Raises:
            RuntimeError: if an index within `indices` is not between [0, itos.size()].
        """
        return self.vocab.lookup_tokens(indices)

    @torch.jit.export
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens (List[str]): the tokens used to lookup their corresponding `indices`.

        Returns:
            indices (List[int]): the 'indices` associated with `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    @torch.jit.export
    def get_stoi(self) -> Dict[str, int]:
        r"""
        Returns:
            stoi (dict): dictionary mapping tokens to indices.
        """
        return self.vocab.get_stoi()

    @torch.jit.export
    def get_itos(self) -> List[str]:
        r"""
        Returns:
            stoi (dict): dictionary mapping indices to tokens.
        """
        return self.vocab.get_itos()
