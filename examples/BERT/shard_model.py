import torch.nn as nn
from model import TransformerEncoderLayer, XLMREmbedding

class XLMRModelShards():
    def __init__(self, ntoken, ninp, nhead, nhid, dropout=0.5):
        self.ntoken = ntoken
        self.ninp = ninp
        self.dropout = dropout
        self.encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)

    def xlmr_embed(self):
        return XLMREmbedding(self.ntoken, self.ninp, self.dropout)

    def encoder_layers(self, nlayers):
        return nn.TransformerEncoder(self.encoder_layer, nlayers)



class MLMShards():
    def __init__(self, ntoken, ninp):
        self.ntoken = ntoken
        self.ninp = ninp

    def mlm(self):
        return nn.Sequential(
            nn.Linear(self.ninp, self.ninp),
            nn.GELU(),
            nn.LayerNorm(self.ninp, eps=1e-12),
            nn.Linear(self.ninp, self.ntoken)
        )
