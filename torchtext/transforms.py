from . import functional as F
from torch.nn import Module
from torch import Tensor
import torch
from torchtext.data.functional import load_sp_model
from torchtext.utils import download_from_url
from torchtext.vocab import Vocab
from typing import List, Optional, Union
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
    Transform for Sentence Piece tokenizer from pre-trained SentencePiece model

    Examples:
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

    def forward(self, input: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(input, list):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsPieces(text))
            return tokens
        else:
            return self.sp_model.EncodeAsPieces(input)


class VocabTransform(Module):
    r"""Vocab transform

    Args:
        vocab: an instance of torchtext.vocab.Vocab class.

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

    def __init__(self, vocab):
        super().__init__()
        assert isinstance(vocab, Vocab)
        self.vocab = vocab

    def forward(self, input: Union[List[str], List[List[str]]]) -> Union[List[int], List[List[int]]]:
        r"""

        Args:
            input: list of list tokens
        """

        if torch.jit.isinstance(input, List[str]):
            return self.vocab.lookup_indices(input)
        else:
            output: List[List[int]] = []
            for tokens in input:
                output.append(self.vocab.lookup_indices(tokens))

            return output


class ToTensor(Module):
    r"""Convert input to torch tensor

    Args:
        padding_value (int, optional): Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    """

    def __init__(self, padding_value: Optional[int] = None) -> None:
        super().__init__()
        self.padding_value = padding_value

    def forward(self, input: List[List[int]]) -> Tensor:
        r"""
        Args:

        """
        return F.to_tensor(input, padding_value=self.padding_value)


class LabelToIndex(Module):
    r"""
    Transform labels from string names to ids.

    Args:
        label_names (List[str], Optional): a list of unique label names
        label_path (str, Optional): a path to file containing unique label names containing 1 label per line.
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
        self._label_vocab = Vocab(torch.classes.torchtext.Vocab(label_names, 0))
        self._label_names = self._label_vocab.get_itos()

    def forward(self, labels: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(labels, list):
            return self._label_vocab.lookup_indices(labels)
        else:
            return self._label_vocab.__getitem__(labels)

    @property
    def label_names(self) -> List[str]:
        return self._label_names
