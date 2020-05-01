import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def init_weights(self):
        self.pos_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        S, N = x.size()
        pos = torch.arange(S,
                           dtype=torch.long,
                           device=x.device).unsqueeze(0).expand((N, S)).t()
        return self.pos_embedding(pos)


class TokenTypeEncoding(nn.Module):
    def __init__(self, type_token_num, d_model):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(type_token_num, d_model)

    def init_weights(self):
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size()
        if token_type_input is None:
            token_type_input = torch.zeros((S, N),
                                           dtype=torch.long,
                                           device=seq_input.device)
        return self.token_type_embeddings(token_type_input)


class MultiheadAttentionInProjection(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None):
        super(MultiheadAttentionInProjection, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.fc_q = nn.Linear(embed_dim, embed_dim)  # query
        self.fc_k = nn.Linear(embed_dim, self.kdim)  # key
        self.fc_v = nn.Linear(embed_dim, self.vdim)  # value

    def init_weights(self):
        self.fc_q.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_k.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_v.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, query, key, value):
        tgt_len, bsz, embed_dim = query.size(0), query.size(1), query.size(2)

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        q = self.fc_q(query)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = self.fc_k(key)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = self.fc_v(value)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        return q, k, v


class ScaledDotProduct(nn.Module):
    def __init__(self, dropout=0.0):
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, attn_mask=None):
        attn_output_weights = torch.bmm(query, key.transpose(1, 2))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(attn_output_weights,
                                                    p=self.dropout,
                                                    training=self.training)
        attn_output = torch.bmm(attn_output_weights, value)
        return attn_output


class MultiheadAttentionOutProjection(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionOutProjection, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = nn.Linear(embed_dim, embed_dim)

    def init_weights(self):
        self.linear.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, attn_output):
        batch_heads, tgt_len = attn_output.size(0), attn_output.size(1)
        bsz = batch_heads // self.num_heads
        assert bsz * self.num_heads == batch_heads, \
            "batch size times the number of heads not equal to attn_output[0]"
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len,
                                                                    bsz,
                                                                    self.embed_dim)
        return self.linear(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.attn_in_proj = MultiheadAttentionInProjection(d_model, nhead)
        self.scaled_dot_product = ScaledDotProduct(dropout=dropout)
        self.attn_out_proj = MultiheadAttentionOutProjection(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def init_weights(self):
        self.attn_in_proj.init_weights()
        self.attn_out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        query, key, value = self.attn_in_proj(src, src, src)
        attn_out = self.scaled_dot_product(query, key, value, attn_mask=src_mask)
        src2 = self.attn_out_proj(attn_out)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def init_weights(self):
        for mod in self.layers:
            mod.init_weights()

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)  # Two sentence type
        self.norm = LayerNorm(ninp)
        self.dropout = Dropout(dropout)

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.pos_embed.init_weights()
        self.tok_type_embed.init_weights()

    def forward(self, src, token_type_input):
        src = self.embed(src) + self.pos_embed(src) \
            + self.tok_type_embed(src, token_type_input)
        return self.dropout(self.norm(src))


class BertModel(nn.Module):
    """Contain a transformer encoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(BertModel, self).__init__()
        self.model_type = 'Transformer'
        self.bert_embed = BertEmbedding(ntoken, ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        self.bert_embed.init_weights()
        self.transformer_encoder.init_weights()

    def forward(self, src, token_type_input):
        src = self.bert_embed(src, token_type_input)
        output = self.transformer_encoder(src)
        return output


class MLMTask(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MLMTask, self).__init__()
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.mlm_span = nn.Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = torch.nn.LayerNorm(ninp, eps=1e-12)
        self.mlm_head = nn.Linear(ninp, ntoken)

    def forward(self, src, token_type_input=None):
        src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model(src, token_type_input)
        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class NextSentenceTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, bert_model):
        super(NextSentenceTask, self).__init__()
        self.bert_model = bert_model
        self.linear_layer = nn.Linear(bert_model.ninp,
                                      bert_model.ninp)
        self.ns_span = nn.Linear(bert_model.ninp, 2)
        self.activation = nn.Tanh()

    def forward(self, src, token_type_input):
        src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model(src, token_type_input)

        # Send the first <'cls'> seq to a classifier
        output = self.activation(self.linear_layer(output[0]))
        output = self.ns_span(output)
        return output


class QuestionAnswerTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, bert_model):
        super(QuestionAnswerTask, self).__init__()
        self.bert_model = bert_model
        self.activation = F.gelu
        self.qa_span = nn.Linear(bert_model.ninp, 2)

    def forward(self, src, token_type_input):
        output = self.bert_model(src, token_type_input)
        # transpose output (S, N, E) to (N, S, E)
        output = output.transpose(0, 1)
        output = self.activation(output)
        pos_output = self.qa_span(output)
        start_pos, end_pos = pos_output.split(1, dim=-1)
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)
        return start_pos, end_pos
