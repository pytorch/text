import torch
import torch.nn as nn
from typing import List
from torchtext.utils import download_from_url
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind
from collections import OrderedDict
from torch import Tensor
from torchtext._torchtext import SentencePiece as SentencePieceCpp


__all__ = [
    'basic_english_normalize',
    'regex_tokenizer',
    'BasicEnglishNormalize',
    'RegexTokenizer',
    'TextSequentialTransforms',
    'load_pretrained_sp_model',
    'sentencepiece_tokenizer',
    'SentencePieceTokenizer',
    'sentencepiece_processor',
    'SentencePieceProcessor',
    'VocabTransform',
    'VectorTransform'
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
    __ignored_properties__ = ["is_jitable"]
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
            lines (str): a text string to tokenize.

        Returns:
            List[str]: a token list after normalizing and splitting on whitespace.
        """
        return self.regex_tokenizer.forward(line)

    def to_ivalue(self):
        r"""Return a JITable BasicEnglishNormalize.
        """
        regex_tokenizer = torch.classes.torchtext.RegexTokenizer(self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, True)
        return BasicEnglishNormalize(regex_tokenizer)


class RegexTokenizer(nn.Module):
    __ignored_properties__ = ["is_jitable"]
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
            lines (str): a text string to tokenize.

        Returns:
            List[str]: a token list after regex.
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
            >>> from torchtext.experimental.transforms import basic_english_normalize, TextSequentialTransforms
            >>> tokenizer = basic_english_normalize()
            >>> txt_pipeline = TextSequentialTransforms(tokenizer)
            >>> txt_pipeline('here is an example')
                ['here', 'is', 'an', 'example']
            >>> jit_txt_pipeline = torch.jit.script(txt_pipeline.to_ivalue())
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


_pretrained_spm = ['text_unigram_15000', 'text_unigram_25000', 'text_unigram_50000',
                   'text_bpe_15000', 'text_bpe_25000', 'text_bpe_50000']


def load_pretrained_sp_model(spm_model='text_unigram_25000'):
    r"""Generate a pretrained sentencepiece model.
        The model was trained with torchtext.datasets.WikiText103, torchtext.datasets.EnWik9 and BookCorpus.
        Both BPE and unigram methods were used to train the model (for more details please refer to
        SentencePiece GitHub https://github.com/google/sentencepiece). We also provide the pretrained model
        with a different size of the vocabulary (i.e. 15000, 25000, 50000).

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
        >>> from torchtext.experimental.transforms import load_pretrained_sp_model
        >>> sp_model = load_pretrained_sp_model('text_unigram_25000')

    """
    if spm_model in _pretrained_spm:
        spm_model = download_from_url('https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/{}.model'.format(spm_model))
        with open(spm_model, 'rb') as f:
            # return SentencePieceCpp(f.read())
            return torch.classes.torchtext.SentencePiece(f.read())
    else:
        raise RuntimeError('The pretrained sentencepiece model is not valid')


def sentencepiece_tokenizer(spm_model):
    r"""Factory function to generate SentencePieceTokenizer from a pretrained SentencePiece model

    Args:
       spm_model: the sentencepiece model instance

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import SentencePieceTokenizer
        >>> from torchtext.experimental.transforms import load_pretrained_sp_model
        >>> sp_model = load_pretrained_sp_model('text_unigram_25000')
        >>> spm_tokenizer = sentencepiece_tokenizer(sp_model)
        >>> jit_spm_tokenizer = torch.jit.script(spm_tokenizer.to_ivalue())
    """
    return SentencePieceTokenizer(spm_model)


class SentencePieceTokenizer(nn.Module):
    r"""Tokenizer based on a pretained sentencepiece model.

    Args:
       spm_model: the sentencepiece model instance
    """

    def __init__(self, spm_model):
        super(SentencePieceTokenizer, self).__init__()
        self.sp_model = spm_model

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line: the input sentence string

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

#    def to_ivalue(self):
#        cpp_spm = torch.classes.torchtext.SentencePiece(bytes(self.sp_model.content_, 'utf-8'))
#        return SentencePieceProcessor(cpp_spm)


def sentencepiece_processor(spm_model):
    r"""Factory function to generate SentencePieceProcessor from a pretrained SentencePiece model

    Args:
       spm_model: the sentencepiece model instance

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import SentencePieceProcessor
        >>> from torchtext.experimental.transforms import load_pretrained_sp_model
        >>> sp_model = load_pretrained_sp_model('text_unigram_25000')
        >>> spm_transform = sentencepiece_processor(sp_model)
        >>> jit_spm_tokenizer = torch.jit.script(spm_transform.to_ivalue())
    """
    return SentencePieceProcessor(spm_model)


class SentencePieceProcessor(nn.Module):
    r"""String to ids transform based on a pretained sentencepiece model

    Args:
       spm_model: the sentencepiece model instance
    """

    def __init__(self, spm_model):
        super(SentencePieceProcessor, self).__init__()
        self.sp_model = spm_model

    def forward(self, line: str) -> List[int]:
        r"""
        Args:
            line: the input sentence string

        Examples:
            >>> spm_processor('the pretrained sp model names')
            >>> [9, 1546, 18811, 2849, 2759, 2202]
        """
        return self.sp_model.EncodeAsIds(line)

    @torch.jit.export
    def decode(self, ids: List[int]) -> str:
        r"""
        Args:
            ids: the integers list for decoder

        Examples:
            >>> spm_processor.decoder([9, 1546, 18811, 2849, 2759, 2202])
            >>> 'the pretrained sp model names'
        """
        return self.sp_model.DecodeIds(ids)

#    def to_ivalue(self):
#        cpp_spm = torch.classes.torchtext.SentencePiece(bytes(self.sp_model.content_, 'utf-8'))
#        print(self.sp_model.content_)
#        cpp_spm = torch.classes.torchtext.SentencePiece(self.sp_model.content_)
#        return SentencePieceProcessor(cpp_spm)


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
            tokens: a string token list

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
            tokens: a string token list

        Example:
            >>> vector_transform(['here', 'is', 'an', 'example'])

        """
        return self.vector.lookup_vectors(tokens)

    def to_ivalue(self):
        if hasattr(self.vector, 'to_ivalue'):
            vector = self.vector.to_ivalue()
            return VectorTransform(vector)
        return self
