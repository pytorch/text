import logging
from typing import Dict, List
from collections import Counter, OrderedDict
import torch
import torch.nn as nn
from torchtext._torchtext import (
    Vocab as VocabPybind,
    _load_vocab_from_file,
    _build_vocab_from_text_file
)

__all__ = [
    'build_vocab_from_text_file',
    'load_vocab_from_file',
    'vocab',
    'Vocab',
]
logger = logging.getLogger(__name__)


def build_vocab_from_text_file(file_object, jited_tokenizer, min_freq=1, num_cpus=4):
    r"""Create a `Vocab` object from a raw text file.

    The `file_object` can contain any raw text. This function applies a generic JITed tokenizer in
    parallel to the text. Note that the vocab will be created in the order that the tokens first appear
    in the file (and not by the frequency of tokens).

    Args:
        file_object (FileObject): a file object to read data from.
        jited_tokenizer (ScriptModule): a tokenizer that has been JITed using `torch.jit.script`
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        num_cpus (int): the number of cpus to use when loading the vectors from file. Default: 4.

    Returns:
        Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.experimental.vocab import build_vocab_from_text_file
        >>> from torchtext.experimental.transforms import basic_english_normalize
        >>> f = open('vocab.txt', 'r')
        >>>     tokenizer = basic_english_normalize()
        >>> tokenizer = basic_english_normalize()
        >>> jit_tokenizer = torch.jit.script(tokenizer.to_ivalue())
        >>> v = build_vocab_from_text_file(f, jit_tokenizer)
        >>> v.insert_token('<unk>', 0)
        >>> v.set_default_index(0)
        >>> v.get_default_index()
    """
    vocab_obj = _build_vocab_from_text_file(file_object.name, min_freq, num_cpus, jited_tokenizer)
    return Vocab(vocab_obj)


def load_vocab_from_file(file_object, min_freq=1, num_cpus=4):
    r"""Create a `Vocab` object from a text file.
    The `file_object` should contain tokens separated by new lines. Note that the vocab
    will be created in the order that the tokens first appear in the file (and not by the frequency of tokens).
    Format for txt file:

        token1
        token2
        ...
        token_n

    Args:
        file_object (FileObject): a file like object to read data from.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        num_cpus (int): the number of cpus to use when loading the vectors from file. Default: 4.

    Returns:
        Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.experimental.vocab import load_vocab_from_file
        >>> f = open('vocab.txt', 'r')
        >>> v = load_vocab_from_file(f)
        >>> v.insert_token('<unk>', 0)
        >>> v.set_default_index(0)
        >>> v.get_default_index()
    """
    vocab_obj = _load_vocab_from_file(file_object.name, min_freq, num_cpus)
    return Vocab(vocab_obj)


def build_vocab_from_iterator(iterator, min_freq=1):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.

    Examples:
        >>> from torchtext.experimental.vocab import build_vocab_from_iterator
        >>> tokens = [['this', 'is', 'an', 'example', 'for', 'vocab']]
        >>> v = build_vocab_from_iterator(tokens)
        >>> v.insert_token('<unk>', 0)
        >>> v.set_default_index(0)
        >>> v.get_default_index()
        >>> tokens_iter = iter([['this', 'is', 'an'], ['example', 'for', 'vocab']])
        >>> v1 = build_vocab_from_iterator(tokens_iter)
    """

    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    word_vocab = vocab(ordered_dict, min_freq=min_freq)
    return word_vocab


def vocab(ordered_dict, min_freq=1):
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Arguments:
        ordered_dict (collections.OrderedDict): object holding the frequencies of each token found in the data.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.

    Examples:
        >>> from torchtext.experimental.vocab import vocab
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> v1.insert_token('<unk>', 0)
        >>> v1.set_default_index(0)
        >>> v1.get_default_index()
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]))
    """
    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)
    return Vocab(VocabPybind(tokens))


class Vocab(nn.Module):
    __jit_unused_properties__ = ["is_jitable"]
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
    def forward(self, tokens: List[str]) -> List[int]:
        r"""Calls the `lookup_indices` method
        Args:
            tokens (List[str]): a list of tokens used to lookup their corresponding `indices`.

        Returns:
            indices (List[int]): the 'indices` associated with a list of tokens`.
        """
        return self.vocab.lookup_indices(tokens)

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
    def __setitem__(self, token: str, index: int) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
            index (int): the index corresponding to the associated token.

        Raises:
            RuntimeError: if `index` not between [0, Vocab.size()] or if token already exists in the vocab.
        """
        self.vocab[token] = index

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
    def set_default_index(self, index: int) -> None:
        r"""
        Args:
            index (int): the unknown index.

        """
        self.vocab.set_default_index(index)

    @torch.jit.export
    def get_default_index(self) -> int:
        r"""
        return:
            index (int): the unknown index.

        """
        return self.vocab.get_default_index()

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
        cpp_vocab = torch.classes.torchtext.Vocab(self.vocab.itos_)
        try:
            cpp_vocab.set_default_index(self.vocab.get_default_index())
            return Vocab(cpp_vocab)
        except RuntimeError:
            return Vocab(cpp_vocab)
