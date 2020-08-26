import torch
import torch.nn as nn
from typing import List
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind
from collections import OrderedDict
from torch import Tensor

__all__ = [
    'BasicEnglishNormalize',
    'RegexTokenizer'
]


def basic_english_normalize():
    r"""Basic normalization for a string sentence.

    Normalization includes
        - lowercasing
        - complete some basic text normalization for English words as follows:
            - add spaces before and after '\''
            - remove '\"',
            - add spaces before and after '.'
            - replace '<br \/>'with single space
            - add spaces before and after ','
            - add spaces before and after '('
            - add spaces before and after ')'
            - add spaces before and after '!'
            - add spaces before and after '?'
            - replace ';' with single space
            - replace ':' with single space
            - replace multiple spaces with single space

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import basic_english_normalize
        >>> test_sample = 'Basic English Normalization for a Line of Text'
        >>> basic_eng_norm = basic_english_normalize()
        >>> jit_basic_eng_norm = torch.jit.script(basic_eng_norm.to_ivalue())
        >>> tokens = jit_basic_eng_norm(test_sample)
    """
    patterns_list = [
        (r'\'', ' \'  '),
        (r'\"', ''),
        (r'\.', ' . '),
        (r'<br \/>', ' '),
        (r',', ' , '),
        (r'\(', ' ( '),
        (r'\)', ' ) '),
        (r'\!', ' ! '),
        (r'\?', ' ? '),
        (r'\;', ' '),
        (r'\:', ' '),
        (r'\s+', ' ')]

    patterns = [pair[0] for pair in patterns_list]
    replacements = [pair[1] for pair in patterns_list]
    return BasicEnglishNormalize(RegexTokenizerPybind(patterns, replacements, True))


def regex_tokenizer(patterns_list):
    r"""Regex tokenizer for a string sentence that applies all regex replacements defined in patterns_list.

    Args:
        patterns_list (List[Tuple[str, str]]): a list of tuples (ordered pairs) which contain the regex pattern string
        as the first element and the replacement string as the second element.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import regex_tokenizer
        >>> test_sample = 'Basic Regex Tokenization for a Line of Text'
        >>> patterns_list = [
            (r'\'', ' \'  '),
            (r'\"', '')]
        >>> reg_tokenizer = regex_tokenizer(patterns_list)
        >>> jit_reg_tokenizer = torch.jit.script(reg_tokenizer)
        >>> tokens = jit_reg_tokenizer(test_sample)
    """
    patterns = [pair[0] for pair in patterns_list]
    replacements = [pair[1] for pair in patterns_list]
    return RegexTokenizer(RegexTokenizerPybind(patterns, replacements, False))


class BasicEnglishNormalize(nn.Module):
    r"""Basic normalization for a string sentence.

    Args:
        regex_tokenizer (torch.classes.torchtext.RegexTokenizer or torchtext._torchtext.RegexTokenizer): a cpp regex tokenizer object.
    """
    def __init__(self, regex_tokenizer):
        super(BasicEnglishNormalize, self).__init__()
        self.regex_tokenizer = regex_tokenizer

    @property
    def is_jitable(self):
        return not isinstance(self.regex_tokenizer, RegexTokenizerPybind)

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """
        return self.regex_tokenizer.forward(line)

    def to_ivalue(self):
        r"""Return a JITable BasicEnglishNormalize.
        """
        regex_tokenizer = torch.classes.torchtext.RegexTokenizer(self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, True)
        return BasicEnglishNormalize(regex_tokenizer)


class RegexTokenizer(nn.Module):
    r"""Regex tokenizer for a string sentence that applies all regex replacements defined in patterns_list.

    Args:
        regex_tokenizer (torch.classes.torchtext.RegexTokenizer or torchtext._torchtext.RegexTokenizer): a cpp regex tokenizer object.
    """
    def __init__(self, regex_tokenzier):
        super(RegexTokenizer, self).__init__()
        self.regex_tokenizer = regex_tokenzier

    @property
    def is_jitable(self):
        return not isinstance(self.regex_tokenizer, RegexTokenizerPybind)

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """
        return self.regex_tokenizer.forward(line)

    def to_ivalue(self):
        r"""Return a JITable RegexTokenizer.
        """
        regex_tokenizer = torch.classes.torchtext.RegexTokenizer(self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, False)
        return RegexTokenizer(regex_tokenizer)


class TextSequentialTransforms(nn.Sequential):
    r"""A container to host a sequential text transforms.

        Example:
            >>> import torch
            >>> from torchtext.experimental.transforms import BasicEnglishNormalize, TextSequentialTransforms
            >>> tokenizer = BasicEnglishNormalize()
            >>> txt_pipeline = TextSequentialTransforms(tokenizer)
            >>> jit_txt_pipeline = torch.jit.script(txt_pipeline)
    """
    def forward(self, input: str):
        for module in self:
            input = module(input)
        return input

    def to_ivalue(self):
        r"""Return a JITable TextSequentialTransforms.
        """
        module_list = []
        for _idx, _module in enumerate(self):
            if hasattr(_module, 'to_ivalue'):
                _module = _module.to_ivalue()
            module_list.append((str(_idx), _module))
        return TextSequentialTransforms(OrderedDict(module_list))


class VocabTransform(nn.Module):
    r"""Vocab transform

    Args:
        vocab: an instance of torchtext.experimental.vocab.Vocab class.

    Example:
        >>> import torch
        >>> from torchtext.experimental.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> vocab_transform = VocabTransform(vocab_from_file_object(f))
        >>> jit_vocab_transform = torch.jit.script(vocab_transform.to_ivalue())
    """

    def __init__(self, vocab):
        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        r"""

        Args:
            tokens: a list of string tokens

        Example:
            >>> vocab_transform(['here', 'is', 'an', 'example'])

        """
        return self.vocab.lookup_indices(tokens)

    def to_ivalue(self):
        if hasattr(self.vocab, 'to_ivalue'):
            vocab = self.vocab.to_ivalue()
            return VocabTransform(vocab)
        return self


class VectorTransform(nn.Module):
    r"""Vector transform

    Args:
        vector: an instance of torchtext.experimental.vectors.Vectors class.

    Example:
        >>> import torch
        >>> from torchtext.experimental.vectors import FastText
        >>> vector_transform = VectorTransform(FastText())
        >>> jit_vector_transform = torch.jit.script(vector_transform.to_ivalue())
    """

    def __init__(self, vector):
        super(VectorTransform, self).__init__()
        self.vector = vector

    def forward(self, tokens: List[str]) -> Tensor:
        r"""

        Args:
            tokens: a list of string tokens

        Example:
            >>> vector_transform(['here', 'is', 'an', 'example'])

        """
        return self.vector.lookup_vectors(tokens)

    def to_ivalue(self):
        if hasattr(self.vector, 'to_ivalue'):
            vector = self.vector.to_ivalue()
            return VectorTransform(vector)
        return self
