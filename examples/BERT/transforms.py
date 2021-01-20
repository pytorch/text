import torch.nn as nn
from torchtext.experimental.vocab import vocab
from torchtext.experimental.transforms import load_sp_model
from typing import List
from collections import OrderedDict


class PretrainedSPVocab(nn.Module):
    r"""Vocab based on a pretained sentencepiece model
    """

    def __init__(self, sp_model_path):
        super(PretrainedSPVocab, self).__init__()
        self.sp_model = load_sp_model(sp_model_path)
        unk_id = self.sp_model.unk_id()
        unk_token = self.sp_model.IdToPiece(unk_id)
        vocab_list = [self.sp_model.IdToPiece(i) for i in range(self.sp_model.GetPieceSize())]
        self.vocab = vocab(OrderedDict([(token, 1) for token in vocab_list]), unk_token=unk_token)

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)

    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]
