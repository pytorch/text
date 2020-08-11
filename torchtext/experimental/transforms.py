import torch
import torch.nn as nn
from typing import List, Tuple
from torchtext.data.functional import load_sp_model
from torchtext.utils import download_from_url


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


_pretrained_spm = ['text_unigram_15000', 'text_unigram_25000', 'text_unigram_50000',
                   'text_bpe_15000', 'text_bpe_25000', 'text_bpe_50000']


class PretrainedSPTokenizer(nn.Module):
    r"""Tokenizer based on a pretained sentencepiece model

    Args:
       spm_model: the pretrained spm model names. Default: 'text_unigram_25000'. The following pretrained spm models are provided:
            - text_unigram_15000
            - text_unigram_25000
            - text_unigram_50000
            - text_bpe_15000
            - text_bpe_25000
            - text_bpe_50000

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import PretrainedSPTokenizer
        >>> spm_tokenizer = PretrainedSPTokenizer()
        >>> spm_tokenizer('the pretrained spm model names')
        >>> ['▁the', '▁pre', 'trained', '▁sp', 'm', '▁model', '▁names']
        >>> jit_spm_tokenizer = torch.jit.script(spm_tokenizer)
    """

    def __init__(self, spm_model='text_unigram_25000'):
        super(PretrainedSPTokenizer, self).__init__()
        if spm_model in _pretrained_spm:
            spm_file_path = download_from_url('https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/{}.model'.format(spm_model))
        else:
            raise RuntimeError('The pretrained sentencepiece model is not supported')
        self.sp_model = load_sp_model(spm_file_path)

    def forward(self, line: str) -> List[str]:
        r"""
        """

        return self.sp_model.EncodeAsPieces(line)

    def decode(self, tokens: List[str]) -> str:
        r"""
        """

        return self.sp_model.DecodePieces(tokens)
