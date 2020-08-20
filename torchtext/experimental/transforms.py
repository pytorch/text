import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from torchtext.data.functional import load_sp_model
from torchtext.utils import download_from_url
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind
from collections import OrderedDict

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
            >>> from torchtext.experimental.transforms import PretrainedSPTransform, ToLongTensor, TextSequentialTransforms
            >>> spm_transform = PretrainedSPTransform()
            >>> to_tensor = ToLongTensor()
            >>> txt_pipeline = TextSequentialTransforms(spm_transform, to_tensor)
            >>> jit_txt_pipeline = torch.jit.script(txt_pipeline)
    """
    def forward(self, line: str):
        r"""
        Args:
            line: the input string

        Examples:
            >>> txt_pipeline('the pretrained sp model names')
        """
        for module in self:
            line = module(line)
        return line


_pretrained_spm = ['text_unigram_15000', 'text_unigram_25000', 'text_unigram_50000',
                   'text_bpe_15000', 'text_bpe_25000', 'text_bpe_50000']


class PretrainedSPTokenizer(nn.Module):
    r"""Tokenizer based on a pretained sentencepiece model

    Args:
       spm_model: the pretrained sentencepiece model names. Default: 'text_unigram_25000'. The following pretrained sentencepiece models are provided:
            - text_unigram_15000
            - text_unigram_25000
            - text_unigram_50000
            - text_bpe_15000
            - text_bpe_25000
            - text_bpe_50000
            Otherwise, the file path to the user-provided sentencepiece model is required.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import PretrainedSPTokenizer
        >>> spm_tokenizer = PretrainedSPTokenizer('text_unigram_25000')
        >>> spm_tokenizer = PretrainedSPTokenizer('.data/text_unigram_25000.model')
        >>> jit_spm_tokenizer = torch.jit.script(spm_tokenizer)
    """

    def __init__(self, spm_model='text_unigram_25000'):
        super(PretrainedSPTokenizer, self).__init__()
        if spm_model in _pretrained_spm:
            spm_model = download_from_url('https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/{}.model'.format(spm_model))
        try:
            self.sp_model = load_sp_model(spm_model)
        except:
            raise RuntimeError('The pretrained sentencepiece model is not supported')

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line: the input string

        Examples:
            >>> spm_tokenizer('the pretrained sp model names')
            >>> ['▁the', '▁pre', 'trained', '▁sp', '▁model', '▁names']
        """

        return self.sp_model.EncodeAsPieces(line)

    @torch.jit.export
    def decode(self, tokens: List[str]) -> str:
        r"""
        Args:
            tokens: the tokens list for decoder

        Examples:
            >>> spm_transform.decoder(['▁the', '▁pre', 'trained', '▁sp', '▁model', '▁names'])
            >>> 'the pretrained sp model names'
        """

        return self.sp_model.DecodePieces(tokens)


class PretrainedSPTransform(nn.Module):
    r"""string to ids transform based on a pretained sentencepiece model

    Args:
       spm_model: the pretrained sentencepiece model names. Default: 'text_unigram_25000'. The following pretrained sentencepiece models are provided:
            - text_unigram_15000
            - text_unigram_25000
            - text_unigram_50000
            - text_bpe_15000
            - text_bpe_25000
            - text_bpe_50000
            Otherwise, the file path to the user-provided sentencepiece model is required.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import PretrainedSPTransform
        >>> spm_transform = PretrainedSPTransform('text_unigram_25000')
        >>> spm_transform = PretrainedSPTransform('.data/text_unigram_25000.model')
        >>> jit_spm_transform = torch.jit.script(spm_transform)
    """

    def __init__(self, spm_model='text_unigram_25000'):
        super(PretrainedSPTransform, self).__init__()
        if spm_model in _pretrained_spm:
            spm_model = download_from_url('https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/{}.model'.format(spm_model))
        try:
            self.sp_model = load_sp_model(spm_model)
        except:
            raise RuntimeError('The pretrained sentencepiece model is not supported')

    def forward(self, line: str) -> List[int]:
        r"""
        Args:
            line: the input string

        Examples:
            >>> spm_transform('the pretrained sp model names')
            >>> [9, 1546, 18811, 2849, 2759, 2202]
        """

        return self.sp_model.EncodeAsIds(line)

    @torch.jit.export
    def decode(self, ids: List[int]) -> str:
        r"""
        Args:
            ids: the integer list for decoder

        Examples:
            >>> spm_transform.decoder([9, 1546, 18811, 2849, 2759, 2202])
            >>> 'the pretrained sp model names'
        """

        return self.sp_model.DecodeIds(ids)


class ToLongTensor(nn.Module):
    r"""Convert a list of integers to long tensor

    Examples:
        >>> from torchtext.experimental.transforms import ToLongTensor
        >>> to_tensor = ToLongTensor()
    """

    def __init__(self):
        super(ToLongTensor, self).__init__()

    def forward(self, ids: List[int]) -> Tensor:
        r"""
        Args:
            ids: the input list of ids

        Examples:
            >>> to_tensor([9, 1546, 18811, 2849, 61, 2759, 2202])
            >>> tensor([    9,  1546, 18811,  2849,    61,  2759,  2202])
        """
        return torch.tensor(ids).to(torch.long)

    def to_ivalue(self):
        r"""Return a JITable TextSequentialTransforms.
        """
        module_list = []
        for _idx, _module in enumerate(self):
            if hasattr(_module, 'to_ivalue'):
                _module = _module.to_ivalue()
            module_list.append((str(_idx), _module))
        return TextSequentialTransforms(OrderedDict(module_list))
