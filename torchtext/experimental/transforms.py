import torch
import torch.nn as nn
from typing import List, Tuple, Optional
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

    def forward(self, lines: List[str]) -> List[List[str]]:
        r"""
        Args:
            lines (List[str]): a list of text to tokenize.

        Returns:
            List[List[str]]: a list of token list after normalizing and splitting on whitespace.
        """
        tokens: List[List[str]] = []
        for line in lines:
            tokens.append(self.regex_tokenizer.forward(line))
        return tokens

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

    def forward(self, lines: List[str]) -> List[List[str]]:
        r"""
        Args:
            lines (List[str]): a list of text to tokenize.

        Returns:
            List[List[str]]: a list of token list after normalizing and splitting on whitespace.
        """
        tokens: List[List[str]] = []
        for line in lines:
            tokens.append(self.regex_tokenizer.forward(line))
        return tokens

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
    def forward(self, input: List[str]):
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

    def forward(self, tokens_list: List[List[str]]) -> List[List[int]]:
        r"""

        Args:
            tokens: a list of string token list

        Example:
            >>> vocab_transform([['here', 'is', 'an', 'example']])

        """
        ids: List[List[int]] = []
        for tokens in tokens_list:
            ids.append(self.vocab.lookup_indices(tokens))
        return ids

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

    def forward(self, tokens_list: List[List[str]]) -> List[Tensor]:
        r"""

        Args:
            tokens: a list of string token list

        Example:
            >>> vector_transform([['here', 'is', 'an', 'example']])

        """
        vectors: List[Tensor] = []
        for tokens in tokens_list:
            vectors.append(self.vector.lookup_vectors(tokens))
        return vectors

    def to_ivalue(self):
        if hasattr(self.vector, 'to_ivalue'):
            vector = self.vector.to_ivalue()
            return VectorTransform(vector)
        return self


class PadTransform(nn.Module):
    r"""Pad sequences. Add padding id to the end of a sequences such that the sequences will
        have the same length.

    Args:
        pad_id: the id for padding token
        eos_token_id: the token id for the end-of-sentence mark (Default: None)
        return_key_padding_mask: flag to output key_padding_mask as output (Defualt: True)

    Example:
        >>> pad_id = 2
        >>> pad_transform = PadTransform(pad_id)
        >>> seq_batch = [[5, 4, 5, 6, 7], [1, 3], [7, 5, 8]]
        >>> pad_seq, key_padding_mask = pad_transform(seq_batch)
        >>> jit_pad_transform = torch.jit.script(pad_transform)
    """

    def __init__(self, pad_id, eos_token_id=None, return_key_padding_mask=True):
        super(PadTransform, self).__init__()
        self.pad_id = pad_id
        self.eos_token_id = eos_token_id
        self.return_key_padding_mask = return_key_padding_mask

    @torch.jit.export
    def forward(self, seq_batch: List[List[int]]) -> Tuple[torch.Tensor, Optional[Tensor]]:
        r"""Pad a list of integer lists. The individual integer list might have different length
        such that the padding function will add padding id to the end of list for the same length.

        Args:
            seq_batch: a list of integer lists. Type: List[List[int]]

        Outputs:
            padded_sequence, key_padding_mask Type: Tuple[torch.Tensor, Optional[Tensor]]

        Note:
            The key_padding_mask tensor has the same shape as the output padded_sequence, with a value of False in
            the position of non-pad values and a value of True in the position of pads.
        """
        max_seq_len = max([len(seq) for seq in seq_batch] + [0])
        if self.eos_token_id is not None:
            max_seq_len += 1
        key_padding_mask = torch.zeros(len(seq_batch), max_seq_len)
        output_batch = torch.ones(len(seq_batch), max_seq_len, dtype=torch.long) * self.pad_id
        for idx, seq in enumerate(seq_batch):
            if self.eos_token_id is not None:
                seq = seq + [self.eos_token_id]
            output_batch[idx][:len(seq)] = torch.tensor(seq, dtype=torch.long)
            key_padding_mask[idx][len(seq):] = 1.0
        if self.return_key_padding_mask:
            return output_batch, key_padding_mask.to(torch.bool)
        else:
            return output_batch, None
