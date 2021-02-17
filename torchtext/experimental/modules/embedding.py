import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=514):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        N, S = x.size()
        pos = torch.arange(S, dtype=torch.long,
                           device=x.device).unsqueeze(0).expand((N, S))
        return self.pos_embedding(pos)


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, embed_dim=768, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.ntoken = ntoken
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.embed = nn.Embedding(ntoken, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, src):
        src = self.embed(src) + self.pos_embed(src)
        return self.dropout(self.norm(src))
