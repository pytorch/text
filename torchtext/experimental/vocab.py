import logging
from typing import Dict, List
import warnings

import torch
import torch.nn as nn
from torchtext._torchtext import (
    Vocab as VocabPybind,
    _load_vocab_from_file,
    _load_vocab_from_raw_text_file
)

logger = logging.getLogger(__name__)


def vocab_from_raw_text_file_object(file_like_object, jit_tokenizer, min_freq=1, unk_token='<unk>', num_cpus=1):
    vocab_obj = _load_vocab_from_raw_text_file(file_like_object.name, unk_token, min_freq, num_cpus, jit_tokenizer)
    return Vocab(vocab_obj)


def vocab_from_file_object(file_like_object, min_freq=1, unk_token='<unk>', num_cpus=1):
    r"""Create a `Vocab` object from a file like object.

    The `file_like_object` should contain tokens seperated by new lines. Note that the vocab
    will be created in the order that the tokens first appear in the file (and not by the frequency of tokens).

    Format for txt file:
        token1
        token2
        ...
        token_n

    Args:
        file_like_object (FileObject): a file like object to read data from.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        unk_token: The default unknown token to use. Default: '<unk>'.
        num_cpus (int): the number of cpus to use when loading the vectors from file. Default: 10.

    Returns:
        Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.experimental.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> v = vocab_from_file_object(f)
    """
    vocab_obj = _load_vocab_from_file(file_like_object.name, unk_token, min_freq, num_cpus)
    return Vocab(vocab_obj)


def vocab(ordered_dict, min_freq=1, unk_token='<unk>'):
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.
    Additionally, the if the `unk_token` isn't found inside of the `ordered_dict`, it will be added to the end of the vocab.

    Arguments:
        ordered_dict (collections.OrderedDict): object holding the frequencies of each token found in the data.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        unk_token: The default unknown token to use. Default: '<unk>'.

    Raises:
        ValueError: if a default `unk_token` isn't provided.

    Examples:
        >>> from torchtext.experimental.vocab import vocab
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]))
    """
    if not unk_token:
        raise ValueError("A default unk token wasn't provided.")

    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if unk_token not in tokens:
        tokens.append(unk_token)
        warnings.warn("The `unk_token` '{}' wasn't found in the `ordered_dict`. Adding the `unk_token` "
                      "to the end of the Vocab.".format(unk_token), RuntimeWarning)

    return Vocab(VocabPybind(tokens, unk_token))


class Vocab(nn.Module):
    r"""Creates a vocab object which maps tokens to indices.

    Arguments:
        vocab (torch.classes.torchtext.Vocab or torchtext._torchtext.Vocab): a cpp vocab object.
    """

    def __init__(self, vocab):
        super(Vocab, self).__init__()
        self.vocab = vocab

    @property
    def is_jitable(self):
        return not isinstance(self.vocab, VocabPybind)

    @torch.jit.export
    def __len__(self) -> int:
        r"""
        Returns:
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
            itos (dict): dictionary mapping indices to tokens.
        """
        return self.vocab.get_itos()

    def to_ivalue(self):
        r"""Return a JITable Vocab.
        """
        cpp_vocab = torch.classes.torchtext.Vocab(self.vocab.itos_, self.vocab.unk_token_)
        return Vocab(cpp_vocab)
