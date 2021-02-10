import torch.nn as nn
from torchtext.experimental.modules import TransformerEncoder
from .xlmr_transform import load_xlmr_transform
from .utils import load_state_dict_from_url


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
    encoder = TransformerEncoder(250002, embed_dim=768, nhead=12, feedforward_dim=3072, nlayers=12, dropout=0.2)
    encoder.load_state_dict(load_state_dict_from_url(PRETRAINED['xlmr.base']))
    return encoder, load_xlmr_transform()


def xlmr_regular():
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_regular
        >>> xlmr_regular_model, xlmr_regular_transform = xlmr_regular()
        >>> xlmr_regular_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    encoder.load_state_dict(load_state_dict_from_url(PRETRAINED['xlmr.regular']))
    encoder = TransformerEncoder(250002, embed_dim=1024, nhead=16, feedforward_dim=4096, nlayers=24, dropout=0.2)
    return encoder, load_xlmr_transform()


PRETRAINED = {'xlmr.regular': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_regular-257d1221.pt',
              'xlmr.base': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_base-1c6e095d.pt'}
SHA256 = {'xlmr.regular': '257d1221b310dbc30e58ecdbf39a4126e78ff894befb180bc7658954b4b55dc3',
          'xlmr.base': '1c6e095df239687682107c44cbb3f35e6329a5609a9fc8be9464eb6ad3919fcd'}

##################################################################################
# This part will be moved to stl-text/models folder


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


def _load_sentence_classifier():
    classifier = SentenceClassificationHead(10, embed_dim=768, dropout=0.2)
    classifier.load_state_dict(load_state_dict_from_url(TASK_PRETRAINED['xlmr_base_sentence_classifier']))
    return classifier


class TransformerEncoderSentenceClassification(nn.Module):
    def __init__(self, transformer_encoder, classifier_head):
        super(TransformerEncoderSentenceClassification, self).__init__()
        self.transformer_encoder = transformer_encoder
        self.classifier_head = classifier_head

    def forward(src):
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
    xlmr_transform = load_xlmr_transform()

    # Load classifier head
    sentence_classifier = _load_sentence_classifier()
    return TransformerEncoderSentenceClassification(xlmr_model, sentence_classifier), xlmr_transform


TASK_PRETRAINED = {'xlmr_base_sentence_classifier': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_base_sentence_classifier-7e3fbb3f.pt'}
TASK_SHA256 = {'xlmr_base_sentence_classifier': '7e3fbb3fac705df2be377d9e1cc198ce3a578172a17b1943e94fa2efe592f278'}
