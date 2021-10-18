from torch.nn import Module
from torchtext.data.functional import load_sp_model
from torchtext.utils import download_from_url
import torchtext
from typing import List

import os


class SpmTokenizerTransform(Module):
    """
    Transform for Sentence Piece tokenizer.

    Examples:
        >>> from torchtext.transforms import PRETRAINED_SP_MODEL
        >>> from torchtext.transforms import SpmTokenizerTransform
        >>> transform = SpmTokenizerTransform(PRETRAINED_SP_MODEL["text_unigram_15000"])
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str):
        super().__init__()
        if os.path.exists(sp_model_path):
            local_path = sp_model_path
        else:
            local_path = download_from_url(sp_model_path)
        self.sp_model = load_sp_model(local_path)

    def forward(self, input: List[str]) -> List[List[str]]:
        tokens: List[List[str]] = []
        for text in input:
            tokens.append(self.sp_model.EncodeAsPieces(text))
        return tokens


class VocabTransform(Module):
    r"""Vocab transform

    Args:
        vocab: an instance of torchtext.vocab.Vocab class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> vocab_transform = VocabTransform(vocab_from_file_object(f))
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab):
        super().__init__()
        assert isinstance(vocab, torchtext.vocab.Vocab)
        self.vocab = vocab

    def forward(self, input: List[List[str]]) -> List[List[int]]:
        r"""

        Args:
            input: list of tokens

        Example:
            >>> vocab_transform(['here', 'is', 'an', 'example'])

        """

        output: List[List[int]] = []
        for tokens in input:
            output.append(self.vocab.lookup_indices(tokens))

        return output
