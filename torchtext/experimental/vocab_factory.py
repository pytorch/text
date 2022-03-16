from typing import Callable, Optional

import torch
from torchtext._torchtext import (
    _build_vocab_from_text_file,
    _build_vocab_from_text_file_using_python_tokenizer,
    _load_vocab_from_file,
)
from torchtext.vocab import Vocab

__all__ = [
    "build_vocab_from_text_file",
    "load_vocab_from_file",
]


def build_vocab_from_text_file(
    file_path: str, tokenizer: Optional[Callable] = None, min_freq: int = 1, num_cpus: Optional[int] = 4
) -> Vocab:
    r"""Create a `Vocab` object from a raw text file.
    The `file_path` can contain any raw text. This function applies a generic JITed tokenizer in
    parallel to the text.

    Args:
        file_object: A file object to read data from.
        tokenizer: A python callable to split input sentence into tokens. It can also be a Jited Module.
            By default, the function will do tokenization based on python split() function.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        num_cpus: the number of cpus to use when loading the vectors from file. It will be ignored when tokenizer is not torch scripted (JIT'd)
    Returns:
        torchtext.vocab.Vocab: a `Vocab` object.
    Examples:
        >>> from torchtext.experimental.vocab_factory import build_vocab_from_text_file
        >>> v = build_vocab_from_text_file('vocab.txt') # using python split function as tokenizer
        >>> #using JIT'd tokenizer
        >>> from torchtext.experimental.transforms import basic_english_normalize
        >>> tokenizer = basic_english_normalize()
        >>> tokenizer = basic_english_normalize()
        >>> jit_tokenizer = torch.jit.script(tokenizer)
        >>> v = build_vocab_from_text_file('vocab.txt', jit_tokenizer, num_cpus = 4)
    """

    if not tokenizer:

        def tokenizer(x):
            return x.split()

    if isinstance(tokenizer, torch.jit.ScriptModule) or isinstance(tokenizer, torch.jit.ScriptFunction):
        vocab_obj = _build_vocab_from_text_file(file_path, min_freq, num_cpus, tokenizer)
    else:
        vocab_obj = _build_vocab_from_text_file_using_python_tokenizer(file_path, min_freq, tokenizer)
    return Vocab(vocab_obj)


def load_vocab_from_file(file_path: str, min_freq: int = 1, num_cpus: int = 4) -> Vocab:
    r"""Create a `Vocab` object from a text file.
    The `file_path` should contain tokens separated by new lines.
    Format for txt file:

        token1
        token2
        ...
        token_n

    Args:
        file_object: A file like object to read data from.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        num_cpus: the number of cpus to use when loading the vectors from file.

    Returns:
        torchtext.vocab.Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.vocab import load_vocab_from_file
        >>> v = load_vocab_from_file('vocab.txt')
    """

    vocab_obj = _load_vocab_from_file(file_path, min_freq, num_cpus)
    return Vocab(vocab_obj)
