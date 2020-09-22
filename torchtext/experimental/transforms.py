import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind
from collections import OrderedDict
from torch import Tensor

__all__ = [
    'basic_english_normalize',
    'regex_tokenizer',
    'BasicEnglishNormalize',
    'RegexTokenizer',
    'TextSequentialTransforms',
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


class PadTransform(nn.Module):
    r"""Pad sequences. Add padding id to the end of a sequences such that the sequences will
        have the same length.

    Args:
        pad_id: the id for padding token
        return_key_padding_mask: flag if to output padding_mask as output (Default: True)

    Example:
        >>> pad_id = 2
        >>> pad_transform = PadTransform(pad_id)
        >>> seq_batch = [torch.tensor([5, 4, 5, 6, 7]), torch.tensor([1, 3]), torch.tensor([7, 5, 8])]
        >>> pad_seq, padding_mask = pad_transform(seq_batch)
        >>> jit_pad_transform = torch.jit.script(pad_transform)
    """

    def __init__(self, pad_id, return_key_padding_mask=True):
        super(PadTransform, self).__init__()
        self.pad_id = pad_id
        self.return_key_padding_mask = return_key_padding_mask

    @torch.jit.export
    def forward(self, seq_batch: List[Tensor]) -> Tuple[torch.Tensor, Optional[Tensor]]:
        r"""Pad a list of tensors in the dim of (seq_dim, ...). The individual tensor has different length
        (i.e. seq_dim) such that the padding function will add padding id to the end of list for the same length. It
        assumes that the dimensions after seq_dim of the tensors are same. And the tensors have same dtype, which is
        the dtype of output padded tensor.

        Args:
            seq_batch: a list of torch.tensor. Type: List[Tensor]

        Outputs:
            padded_sequence, padding_mask Type: Tuple[torch.Tensor, Optional[Tensor]]

        Note:
            The padding_mask tensor has the same shape [len(seq_batch), max_seq_len], with a value of False in
            the position of non-pad values and a value of True in the position of pads. len(seq_batch) is the number
            of input sequences and max_seq_len is the maximum length of the input sequences.
        """
        max_seq_len = max([seq.size(0) for seq in seq_batch])
        padding_mask = torch.zeros(len(seq_batch), max_seq_len)
        for idx, seq in enumerate(seq_batch):
            padding_mask[idx][seq.size(0):] = 1.0
        output_batch = torch.nn.utils.rnn.pad_sequence(seq_batch, batch_first=True, padding_value=float(self.pad_id))
        if self.return_key_padding_mask:
            return output_batch, padding_mask.to(torch.bool)
        else:
            return output_batch, None
