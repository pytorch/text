import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

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

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size()
        if token_type_input is None:
            token_type_input = torch.zeros((S, N),
                                           dtype=torch.long,
                                           device=seq_input.device)
        return self.token_type_embeddings(token_type_input)


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

    def forward(self, src, token_type_input):
        src = self.bert_embed(src, token_type_input)
        output = self.transformer_encoder(src)
        return output


class MLMTask(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MLMTask, self).__init__()
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.mlm_span = Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12)
        self.mlm_head = Linear(ninp, ntoken)

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
        self.linear_layer = Linear(bert_model.ninp,
                                   bert_model.ninp)
        self.ns_span = Linear(bert_model.ninp, 2)
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
        self.qa_span = Linear(bert_model.ninp, 2)

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
