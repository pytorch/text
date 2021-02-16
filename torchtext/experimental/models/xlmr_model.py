import torch.nn as nn
from .xlmr_transform import load_xlmr_transform
from .utils import load_state_dict_from_url
from torch.nn import Linear, LayerNorm, TransformerEncoder
import torch.nn.functional as F
from torchtext.experimental.modules import BertEmbedding, TransformerEncoderLayer


class XLMRModel(nn.Module):
    """XLM-R model: a transformer encoder + embedding layer."""

    def __init__(self, ntoken, embed_dim, nhead, feedforward_dim, nlayers, dropout=0.5):
        super(XLMRModel, self).__init__()
        self.xlmr_embed = BertEmbedding(ntoken, embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, nhead, feedforward_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_dim = embed_dim

    def forward(self, src):
        src = self.xlmr_embed(src)
        output = self.transformer_encoder(src)
        return output


# [TODO] Add torch.hub support
# [TODO] Download file from manifold
# [TODO] check base model config
def xlmr_base():
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_base
        >>> xlmr_base_model, xlmr_base_transform = xlmr_base()
        >>> xlmr_base_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    encoder = XLMRModel(250002, embed_dim=768, nhead=12, feedforward_dim=3072, nlayers=12, dropout=0.2)
    encoder.load_state_dict(load_state_dict_from_url(PRETRAINED['xlmr.base'], hash_value=SHA256['xlmr.base']))
    return encoder, load_xlmr_transform()


def xlmr_regular():
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_regular
        >>> xlmr_regular_model, xlmr_regular_transform = xlmr_regular()
        >>> xlmr_regular_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    encoder = XLMRModel(250002, embed_dim=1024, nhead=16, feedforward_dim=4096, nlayers=24, dropout=0.2)
    encoder.load_state_dict(load_state_dict_from_url(PRETRAINED['xlmr.regular'], hash_value=SHA256['xlmr.regular']))
    return encoder, load_xlmr_transform()


PRETRAINED = {'xlmr.regular': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_regular-b39c547d.pt',
              'xlmr.base': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_base-dcbe409a.pt'}
SHA256 = {'xlmr.regular': 'b39c547d43acab913ddc5d902996b65ceeaf13e8068a1d766aa4ac3a2104d6c9',
          'xlmr.base': 'dcbe409aff2609843b6c6e37e3e16bbf068ed8d4995c28dd1cbe24ba3b4534e9'}

##################################################################################
# This part will be moved to stl-text/models folder


###########################
# Sentence Classification
###########################
class SentenceClassificationHead(nn.Module):
    """Head for sentence-level classification."""

    def __init__(self, num_labels, embed_dim=768, dropout=0.2):
        super(SentenceClassificationHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, num_labels)
        self.activation = nn.Tanh()

    def forward(self, input_features):
        x = input_features[:, 0, :]  # The first token is reserved for [CLS]
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def sentence_classifier_head():
    classifier = SentenceClassificationHead(10, embed_dim=768, dropout=0.2)
    classifier.load_state_dict(load_state_dict_from_url(TASK_PRETRAINED['xlmr_base_sentence_classifier'],
                               hash_value=TASK_SHA256['xlmr_base_sentence_classifier']))
    return classifier


class TransformerEncoderSentenceClassification(nn.Module):
    def __init__(self, transformer_encoder, classifier_head):
        super(TransformerEncoderSentenceClassification, self).__init__()
        self.transformer_encoder = transformer_encoder
        self.classifier_head = classifier_head

    def forward(self, src):
        raise NotImplementedError("forward func has not been implemented yet.")


def xlmr_base_sentence_classifier():
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_base_sentence_classifier
        >>> xlmr_sentence_classifier_model, xlmr_base_transform = xlmr_base_sentence_classifier()
        >>> xlmr_base_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    # Load pretrained XLM-R
    xlmr_model, xlmr_transform = xlmr_base()

    # Load classifier head
    sentence_classifier = sentence_classifier_head()
    return TransformerEncoderSentenceClassification(xlmr_model, sentence_classifier), xlmr_transform


###########################
# Language Modeling
###########################
class CrossLingualMLMHead(nn.Module):
    """Contain a cross-lingual MLM head."""

    def __init__(self, ntoken, embed_dim):
        super(CrossLingualMLMHead, self).__init__()
        self.mlm_span = Linear(embed_dim, embed_dim)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(embed_dim, eps=1e-12)
        self.mlm_head = Linear(embed_dim, ntoken)

    def forward(self, src):
        output = self.mlm_span(src)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


def cross_lingual_mlm_head():
    classifier = CrossLingualMLMHead(250002, 768)
    # [TODO] Load the weight of LM head
    return classifier


class TransformerEncoderLanguageModeling(nn.Module):
    """Contain a transformer encoder plus LM head."""

    def __init__(self, transformer_encoder, lm_head):
        super(TransformerEncoderLanguageModeling, self).__init__()
        self.transformer_encoder = transformer_encoder
        self.lm_head = lm_head

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.lm_head(output)
        return output


def xlmr_base_cross_lingual_mlm():
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_base_cross_lingual_mlm
        >>> xlmr_lm_model, xlmr_base_transform = xlmr_base_cross_lingual_mlm()
        >>> xlmr_base_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    xlmr_model, xlmr_transform = xlmr_base()

    lm_head = cross_lingual_mlm_head()
    return TransformerEncoderLanguageModeling(xlmr_model, lm_head), xlmr_transform


TASK_PRETRAINED = {'xlmr_base_sentence_classifier': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_base_sentence_classifier-7e3fbb3f.pt'}
TASK_SHA256 = {'xlmr_base_sentence_classifier': '7e3fbb3fac705df2be377d9e1cc198ce3a578172a17b1943e94fa2efe592f278'}
