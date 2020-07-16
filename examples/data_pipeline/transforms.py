import torch.nn as nn
from torchtext.data.functional import load_sp_model
from torchtext.experimental.vocab import Vocab
from typing import List
from collections import OrderedDict


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
        vocab_list = [self.sp_model.IdToPiece(i) for i in range(unk_id + 1, self.sp_model.GetPieceSize())]
        self.vocab = Vocab(OrderedDict([(token, 1) for token in vocab_list]), unk_token=unk_token,
                           specials=tuple([self.sp_model.IdToPiece(i) for i in range(unk_id + 1)]))

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)

    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)
