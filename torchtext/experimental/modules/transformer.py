import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, nhead=12, feedforward_dim=3072,
                 dropout=0.2, activation=F.gelu):
        super(TransformerEncoderLayer, self).__init__()
        in_proj_container = InProjContainer(Linear(embed_dim, embed_dim),
                                            Linear(embed_dim, embed_dim),
                                            Linear(embed_dim, embed_dim))
        self.mha = MultiheadAttentionContainer(nhead, in_proj_container,
                                               ScaledDotProduct(), Linear(embed_dim, embed_dim), batch_first=True)
        self.linear1 = Linear(embed_dim, feedforward_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(feedforward_dim, embed_dim)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation
        # [TODO] Add init_weights()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
