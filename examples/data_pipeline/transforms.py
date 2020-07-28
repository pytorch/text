import torch.nn as nn
from torchtext.data.functional import load_sp_model
from torchtext.experimental.vocab import Vocab
from typing import List
from collections import OrderedDict
import torch
from torch import Tensor


class TextClassificationPipeline(nn.Module):
    r"""Text classification pipeline template
    """

    def __init__(self, label_transform, text_transform):
        super(TextClassificationPipeline, self).__init__()
        self.label_transform = label_transform
        self.text_transform = text_transform

    def forward(self, label_text_tuple):
        return self.label_transform(label_text_tuple[0]), self.text_transform(label_text_tuple[1])


class TextDataPipeline(nn.Module):
    r"""Text data pipeline template
    """

    def __init__(self, tokenizer, vocab):
        super(TextDataPipeline, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab

    @torch.jit.export
    def forward(self, line: str):
        tokens = self.tokenizer(line)
        index = self.vocab(tokens)
        return index


class PretrainedSPTokenizer(nn.Module):
    r"""Tokenizer based on a pretained sentencepiece model
    """

    def __init__(self, spm_file):
        super(PretrainedSPTokenizer, self).__init__()
        self.sp_model = load_sp_model(spm_file)

    def forward(self, line: str) -> List[str]:
        r"""

        """

        return self.sp_model.EncodeAsPieces(line)


class PretrainedSPVocab(nn.Module):
    r"""Vocab based on a pretained sentencepiece model
    """

    def __init__(self, spm_file):
        super(PretrainedSPVocab, self).__init__()
        self.sp_model = load_sp_model(spm_file)
        unk_id = self.sp_model.unk_id()
        unk_token = self.sp_model.IdToPiece(unk_id)
        vocab_list = [self.sp_model.IdToPiece(i) for i in range(self.sp_model.GetPieceSize())]
        self.vocab = Vocab(OrderedDict([(token, 1) for token in vocab_list]), unk_token=unk_token)

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)

    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)


class VocabTransform(nn.Module):
    r"""Vocab transform
    """

    def __init__(self, vocab):
        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)


class PyTextVocabTransform(nn.Module):
    r"""Vocab transform
    """

    def __init__(self, vocab):
        super(PyTextVocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices_1d(tokens)


class VectorTransform(nn.Module):
    r"""Vector transform
    """

    def __init__(self, vector):
        super(VectorTransform, self).__init__()
        self.vector = vector

    def forward(self, tokens: List[str]) -> Tensor:
        return self.vector.lookup_vectors(tokens)


class TextSequential(nn.Sequential):
    r"""Text Sequential pipeline

        Example:
            >>> from transforms import TextSequential
            >>> from torchtext.experimental.transforms import BasicEnglishNormalize
            >>> from transforms import VocabTransform
            >>> tokenizer = BasicEnglishNormalize()
            >>> from torchtext.experimental.vocab import Vocab
            >>> from collections import Counter, OrderedDict
            >>> v = Vocab(OrderedDict([(token, 1) for token in ['e', 'd', 'c', 'b', 'a']]))
            >>> vocab = VocabTransform(v)
            >>> txt_pipeline = TextSequential(tokenizer, vocab)
            >>> import torch
            >>> jit_txt_pipeline = torch.jit.script(txt_pipeline)
    """
    def forward(self, input: str):
        for module in self:
            input = module(input)
        return input
