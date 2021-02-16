import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, nhead=12, feedforward_dim=3072,
                 dropout=0.2, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        in_proj_container = InProjContainer(Linear(embed_dim, embed_dim),
                                            Linear(embed_dim, embed_dim),
                                            Linear(embed_dim, embed_dim))
        self.mha = MultiheadAttentionContainer(nhead, in_proj_container,
                                               ScaledDotProduct(), Linear(embed_dim, embed_dim))
        self.linear1 = Linear(embed_dim, feedforward_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(feedforward_dim, embed_dim)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def init_weights(self):
        self.mha.in_proj_container.query_proj.init_weights()
        self.mha.in_proj_container.key_proj.init_weights()
        self.mha.in_proj_container.value_proj.init_weights()
        self.mha.out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
