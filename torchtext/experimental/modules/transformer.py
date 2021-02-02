import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, ModuleList
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
import copy
from .embedding import PositionalEmbedding


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
        # attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        # src = src + self.dropout1(attn_output)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        # return src
        raise NotImplementedError("forward func has not been implemented yet.")


class TransformerEncoder(nn.Module):
    """Contain a transformer encoder."""

    def __init__(self, ntoken, embed_dim=768, nhead=12, feedforward_dim=3072, nlayers=12, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.model_type = 'Transformer'
        self.transformer_encoder_embedding = self.build_transformer_encoder_embedding(ntoken, embed_dim)
        encoder_layer = self.build_transformer_encoder_layer(embed_dim, nhead, feedforward_dim, dropout)
        self.encoder_layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(nlayers)])
        self.embed_dim = embed_dim

    def forward(self, src):
        # src = self.transformer_encoder_embedding(src)
        # output = self.encoder_layers(src)
        # return output
        raise NotImplementedError("forward func has not been implemented yet.")

    def build_transformer_encoder_embedding(self, ntoken, embed_dim):
        return BertEmbedding(ntoken, embed_dim)

    def build_transformer_encoder_layer(self, embed_dim, nhead, feedforward_dim, dropout):
        return TransformerEncoderLayer(embed_dim, nhead, feedforward_dim, dropout)


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
        # src = self.embed(src) + self.pos_embed(src)
        # return self.dropout(self.norm(src))
        raise NotImplementedError("forward func has not been implemented yet.")
