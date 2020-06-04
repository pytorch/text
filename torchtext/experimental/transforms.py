import torch
import torch.nn as nn
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

    regex_and_replacement_string_pairs: List[Tuple[torch.classes.torchtext.Regex, str]]

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

        regex_objects = map(lambda pattern_tuple: torch.classes.torchtext.Regex(pattern_tuple[0]), patterns_list)
        replacement_strings = map(lambda pattern_tuple: pattern_tuple[1], patterns_list)
        self.regex_and_replacement_string_pairs = list(zip(regex_objects, replacement_strings))

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """

        line = line.lower()
        for regex, replacement_string in self.regex_and_replacement_string_pairs:
            line = regex.Sub(line, replacement_string)
        return line.split()


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
        >>> tokens = jit_regex_tokenizer.forward(test_sample)
    """

    regex_and_replacement_string_pairs: List[Tuple[torch.classes.torchtext.Regex, str]]

    def __init__(self, patterns_list: List[Tuple[str, str]]):
        super(RegexTokenizer, self).__init__()

        regex_objects = map(lambda pattern_tuple: torch.classes.torchtext.Regex(pattern_tuple[0]), patterns_list)
        replacement_strings = map(lambda pattern_tuple: pattern_tuple[1], patterns_list)
        self.regex_and_replacement_string_pairs = list(zip(regex_objects, replacement_strings))

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line (str): a line of text to tokenize.
        Returns:
            List[str]: a list of tokens after normalizing and splitting on whitespace.
        """

        for regex, replacement_string in self.regex_and_replacement_string_pairs:
            line = regex.Sub(line, replacement_string)
        return line.split()
