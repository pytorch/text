import torch
import torch.nn as nn
from typing import List
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind
from torch import Tensor
from torchtext._torchtext import SentencePiece as SentencePiecePybind
import io


__all__ = [
    'basic_english_normalize',
    'regex_tokenizer',
    'BasicEnglishNormalize',
    'RegexTokenizer',
    'TextSequentialTransforms',
    'PRETRAINED_SP_MODEL',
    'load_sp_model',
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
        >>> jit_basic_eng_norm = torch.jit.script(basic_eng_norm)
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
    __jit_unused_properties__ = ["is_jitable"]
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

    def __prepare_scriptable__(self):
        r"""Return a JITable BasicEnglishNormalize.
        """
        if self.is_jitable:
            return BasicEnglishNormalize(self.regex_tokenizer)
        regex_tokenizer = torch.classes.torchtext.RegexTokenizer(self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, True)
        return BasicEnglishNormalize(regex_tokenizer)


class RegexTokenizer(nn.Module):
    __jit_unused_properties__ = ["is_jitable"]
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

    def __prepare_scriptable__(self):
        r"""Return a JITable RegexTokenizer.
        """
        if self.is_jitable:
            return RegexTokenizer(self.regex_tokenizer)

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
            >>> jit_txt_pipeline = torch.jit.script(txt_pipeline)
    """

    def forward(self, input: str):
        for module in self:
            input = module(input)
        return input


PRETRAINED_SP_MODEL = {
    'text_unigram_15000': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_15000.model',
    'text_unigram_25000': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_25000.model',
    'text_unigram_50000': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_50000.model',
    'text_bpe_15000': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_bpe_15000.model',
    'text_bpe_25000': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_bpe_25000.model',
    'text_bpe_50000': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_bpe_50000.model'}


def load_sp_model(sp_model):
    r"""Load a  sentencepiece model for file.

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Outputs:
        output: a SentencePiece model.

    Examples:
        >>> from torchtext.experimental.transforms import load_sp_model
        >>> sp_model = load_sp_model("m_user.model")
        >>> sp_model = load_sp_model(open("m_user.model", 'rb'))

    Note: We also provide several pretrained sentencepiece models. The model was trained with torchtext.datasets.WikiText103,
        torchtext.datasets.EnWik9 and BookCorpus. Both BPE and unigram methods were used to train the model (for more
        details please refer to SentencePiece GitHub https://github.com/google/sentencepiece). We also provide the pretrained model
        with a different size of the vocabulary (i.e. 15000, 25000, 50000).
        The following pretrained sentencepiece models are provided:

            - text_unigram_15000
            - text_unigram_25000
            - text_unigram_50000
            - text_bpe_15000
            - text_bpe_25000
            - text_bpe_50000

    Examples:
        >>> from torchtext.experimental.transforms import PRETRAINED_SP_MODEL
        >>> sp_model_path = torchtext.utils.download_from_url(PRETRAINED_SP_MODEL['text_unigram_25000'])
        >>> sp_model = load_sp_model(sp_model_path)
    """

    if isinstance(sp_model, str):
        with open(sp_model, 'rb') as f:
            return SentencePiecePybind(f.read())
    elif isinstance(sp_model, io.BufferedReader):
        return SentencePiecePybind(sp_model.read())
    else:
        raise TypeError(
            f'Unsupported type for sp_model argument: {type(sp_model).__name__}. ' +
            'Supported types are: ' +
            ', '.join([
                'str', 'io.BufferedReader'
            ]))


def sentencepiece_tokenizer(sp_model):
    r"""Factory function to generate SentencePieceTokenizer from a pretrained SentencePiece model

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import sentencepiece_tokenizer
        >>> spm_tokenizer = sentencepiece_tokenizer('m_user.model')
        >>> jit_spm_tokenizer = torch.jit.script(spm_tokenizer)
    """

    spm = load_sp_model(sp_model)
    return SentencePieceTokenizer(spm)


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

        Note: SentencePiece treats the input text just as a sequence of Unicode characters. Whitespace is also handled as a normal symbol. To handle the whitespace as a basic token explicitly, SentencePiece first escapes the whitespace with a meta symbol "▁" (U+2581) as follows.
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

    def __prepare_scriptable__(self):
        torchbind_spm = torch.classes.torchtext.SentencePiece(self.sp_model._return_content())
        return SentencePieceTokenizer(torchbind_spm)


def sentencepiece_processor(sp_model):
    r"""Factory function to generate SentencePieceProcessor from a pretrained SentencePiece model

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import sentencepiece_processor
        >>> spm_processor = sentencepiece_processor('m_user.model')
        >>> jit_spm_processor = torch.jit.script(spm_processor)
    """

    spm = load_sp_model(sp_model)
    return SentencePieceProcessor(spm)


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

    def __prepare_scriptable__(self):
        torchbind_spm = torch.classes.torchtext.SentencePiece(self.sp_model._return_content())
        return SentencePieceProcessor(torchbind_spm)


class VocabTransform(nn.Module):
    r"""Vocab transform

    Args:
        vocab: an instance of torchtext.experimental.vocab.Vocab class.

    Example:
        >>> import torch
        >>> from torchtext.experimental.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> vocab_transform = VocabTransform(vocab_from_file_object(f))
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
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

    def __prepare_scriptable__(self):
        if hasattr(self.vocab, '__prepare_scriptable__'):
            vocab = self.vocab.__prepare_scriptable__()
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
        >>> jit_vector_transform = torch.jit.script(vector_transform)
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

    def __prepare_scriptable__(self):
        if hasattr(self.vector, '__prepare_scriptable__'):
            vector = self.vector.__prepare_scriptable__()
            return VectorTransform(vector)
        return self
