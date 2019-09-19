import torch
import torch.nn as nn
import sentencepiece as spm


class SentencePieceTransform(nn.Module):
    def __init__(self, spm_filename):
        super(SentencePieceTransform, self).__init__()
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_filename)

    def forward(self, txt_str):
        return self.sp_model.EncodeAsIds(txt_str)
