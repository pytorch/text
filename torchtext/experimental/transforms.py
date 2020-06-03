import torch
import torchtext
import torch.nn as nn
from torchtext.experimental import functional as F
from typing import List, Tuple


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
        >>> tokens = jit_basic_english_normalize.forward(test_sample)
    """

    patterns_list: List[Tuple[str, str]]
    regex: torch.classes.torchtext.Regex

    def __init__(self):
        super(BasicEnglishNormalize, self).__init__()
        self.regex = torch.classes.torchtext.Regex('')
        self.patterns_list = [
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

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """
        return F.regex_tokenizer(line, self.patterns_list, self.regex, lowercase=True)


class RegexTokenizer(nn.Module):
    r"""Regex tokenizer for a string sentence that applies all regex replacements defined in patterns_list.

    Args:
        patterns_list (List[Tuple[str, str]]): a list of tuples (ordered pairs) which contain the regex pattern string as the first element and the replacement string as the second element.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import RegexTokenizer
        >>> test_sample = 'Basic Regex Tokenization for a Line of Text'
        >>> patterns_list = [
            (r'\'', ' \'  '),
            (r'\"', '')]
        >>> regex_tokenizer = RegexTokenizer(patterns_list)
        >>> jit_regex_tokenizer = torch.jit.script(regex_tokenizer)
        >>> tokens = jit_regex_tokenizer.forward(test_sample)
    """

    patterns_list: List[Tuple[str, str]]
    regex: torch.classes.torchtext.Regex

    def __init__(self, patterns_list: List[Tuple[str, str]]):
        super(RegexTokenizer, self).__init__()
        self.regex = torch.classes.torchtext.Regex('')
        self.patterns_list = patterns_list

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """
        return F.regex_tokenizer(line, self.patterns_list, self.regex)
