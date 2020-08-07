import torch
import torch.nn as nn
from typing import List, Tuple

# from torchtext._torchtext import Regex as RegexPybind

__all__ = [
    'BasicEnglishNormalize',
    'RegexTokenizer'
]


class BasicEnglishNormalize(nn.Module):
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
        >>> from torchtext.experimental.transforms import BasicEnglishNormalize
        >>> test_sample = 'Basic English Normalization for a Line of Text'
        >>> basic_english_normalize = BasicEnglishNormalize()
        >>> jit_basic_english_normalize = torch.jit.script(basic_english_normalize)
        >>> tokens = jit_basic_english_normalize(test_sample)
    """
    def __init__(self):
        super(BasicEnglishNormalize, self).__init__()
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
        self.regex_tokenizer = torch.classes.torchtext.RegexTokenizer(patterns, replacements, True)

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """
        return self.regex_tokenizer.forward(line)


class RegexTokenizer(nn.Module):
    r"""Regex tokenizer for a string sentence that applies all regex replacements defined in patterns_list.

    Args:
        patterns_list (List[Tuple[str, str]]): a list of tuples (ordered pairs) which contain the regex pattern string
        as the first element and the replacement string as the second element.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import RegexTokenizer
        >>> test_sample = 'Basic Regex Tokenization for a Line of Text'
        >>> patterns_list = [
            (r'\'', ' \'  '),
            (r'\"', '')]
        >>> regex_tokenizer = RegexTokenizer(patterns_list)
        >>> jit_regex_tokenizer = torch.jit.script(regex_tokenizer)
        >>> tokens = jit_regex_tokenizer(test_sample)
    """
    def __init__(self, patterns_list: List[Tuple[str, str]]):
        super(RegexTokenizer, self).__init__()

        patterns = [pair[0] for pair in patterns_list]
        replacements = [pair[1] for pair in patterns_list]
        self.regex_tokenizer = torch.classes.torchtext.RegexTokenizer(patterns, replacements, False)

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """
        return self.regex_tokenizer.forward(line)


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
