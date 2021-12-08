from . import functional as F
from torch.nn import Module
from torch import Tensor
import torch
from torchtext.data.functional import load_sp_model
from torchtext.utils import download_from_url
from torchtext.vocab import Vocab
from typing import List, Optional, Any
import os

from torchtext import _CACHE_DIR

__all__ = [
    'SentencePieceTokenizer',
    'VocabTransform',
    'ToTensor',
    'LabelToIndex',
]


class SentencePieceTokenizer(Module):
    """
    Transform for Sentence Piece tokenizer from pre-trained sentencepiece model

    Additiona details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str

    Example
        >>> from torchtext.transforms import SpmTokenizerTransform
        >>> transform = SentencePieceTokenizer("spm_model")
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str):
        super().__init__()
        if os.path.exists(sp_model_path):
            local_path = sp_model_path
        else:
            local_path = download_from_url(url=sp_model_path, root=_CACHE_DIR)
        self.sp_model = load_sp_model(local_path)

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsPieces(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsPieces(input)
        else:
            raise TypeError("Input type not supported")


class VocabTransform(Module):
    r"""Vocab transform to convert input batch of tokens into corresponding token ids

    :param vocab: an instance of :class:`torchtext.vocab.Vocab` class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab
        >>> from torchtext.transforms import VocabTransform
        >>> from collections import OrderedDict
        >>> vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        >>> vocab_transform = VocabTransform(vocab_obj)
        >>> output = vocab_transform([['a','b'],['a','b','c']])
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab: Vocab):
        super().__init__()
        assert isinstance(vocab, Vocab)
        self.vocab = vocab

    def forward(self, input: Any) -> Any:
        """
        :param input: Input batch of token to convert to correspnding token ids
        :type input: Union[List[str], List[List[str]]]
        :return: Converted input into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """

        if torch.jit.isinstance(input, List[str]):
            return self.vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, List[List[str]]):
            output: List[List[int]] = []
            for tokens in input:
                output.append(self.vocab.lookup_indices(tokens))

            return output
        else:
            raise TypeError("Input type not supported")


class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: Optional[int] = None, dtype: Optional[torch.dtype] = torch.long) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return F.to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)


class LabelToIndex(Module):
    r"""
    Transform labels from string names to ids.

    :param label_names: a list of unique label names
    :type label_names: Optional[List[str]]
    :param label_path: a path to file containing unique label names containing 1 label per line. Note that either label_names or label_path should be supplied
                       but not both.
    :type label_path: Optional[str]
    """

    def __init__(
        self, label_names: Optional[List[str]] = None, label_path: Optional[str] = None, sort_names=False,
    ):
        assert label_names or label_path, "label_names or label_path is required"
        assert not (label_names and label_path), "label_names and label_path are mutually exclusive"
        super().__init__()

        if label_path:
            with open(label_path, "r") as f:
                label_names = [line.strip() for line in f if line.strip()]
        else:
            label_names = label_names

        if sort_names:
            label_names = sorted(label_names)
        self._label_vocab = Vocab(torch.classes.torchtext.Vocab(label_names, None))
        self._label_names = self._label_vocab.get_itos()

    def forward(self, input: Any) -> Any:
        """
        :param input: Input labels to convert to corresponding ids
        :type input: Union[str, List[str]]
        :rtype: Union[int, List[int]]
        """
        if torch.jit.isinstance(input, List[str]):
            return self._label_vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, str):
            return self._label_vocab.__getitem__(input)
        else:
            raise TypeError("Input type not supported")

    @property
    def label_names(self) -> List[str]:
        return self._label_names


class Truncate(Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.truncate(input, self.max_seq_len)
