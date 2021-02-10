import torch
import torch.nn as nn
from torchtext.experimental.modules import TransformerEncoder
from .xlmr_transform import load_xlmr_transform
from torchtext.utils import download_from_url, extract_archive, load_args_from_json


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
    tar_file = download_from_url(PRETRAINED['xlmr.base'],
                                 hash_value=MD5['xlmr.base'], hash_type='md5')
    checkpoint_file, tokenizer_file, vocab_file, args_file = extract_archive(tar_file, overwrite=True)
    return _load_xlmr_model(checkpoint_file=checkpoint_file, args_file=args_file), load_xlmr_transform(tokenizer_file=tokenizer_file, vocab_file=vocab_file)


def xlmr_regular():
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_regular
        >>> xlmr_regular_model, xlmr_regular_transform = xlmr_regular()
        >>> xlmr_regular_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    tar_file = download_from_url(PRETRAINED['xlmr.regular'],
                                 hash_value=MD5['xlmr.regular'], hash_type='md5')
    checkpoint_file, tokenizer_file, vocab_file, args_file = extract_archive(tar_file, overwrite=True)
    return _load_xlmr_model(checkpoint_file=checkpoint_file, args_file=args_file), load_xlmr_transform(tokenizer_file=tokenizer_file, vocab_file=vocab_file)


def _load_xlmr_model(checkpoint_file='model.pt', args_file='args.json'):
    args = load_args_from_json(args_file)
    encoder = TransformerEncoder(args.ntoken, args.embed_dim, args.nhead,
                                 args.feedforward_dim, args.nlayers, args.dropout)
    encoder.load_state_dict(torch.load(checkpoint_file))
    return encoder


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


def _load_sentence_classifier(checkpoint_file='model.pt', args_file='args.json'):
    args = load_args_from_json(args_file)
    classifier = SentenceClassificationHead(args.num_labels, args.embed_dim, args.dropout)
    classifier.load_state_dict(torch.load(checkpoint_file))
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
    tar_file = download_from_url(PRETRAINED['xlmr.base'],
                                 hash_value=MD5['xlmr.base'], hash_type='md5')
    checkpoint_file, tokenizer_file, vocab_file, args_file = extract_archive(tar_file, overwrite=True)
    xlmr_model = _load_xlmr_model(checkpoint_file=checkpoint_file, args_file=args_file)
    xlmr_transform = load_xlmr_transform(tokenizer_file=tokenizer_file, vocab_file=vocab_file)

    # Load classifier head
    tar_file = download_from_url(TASK_PRETRAINED['xlmr.base.sentence.classifier'],
                                 hash_value=TASK_MD5['xlmr.base.sentence.classifier'], hash_type='md5')
    checkpoint_file, args_file = extract_archive(tar_file, overwrite=True)
    sentence_classifier = _load_sentence_classifier(checkpoint_file=checkpoint_file, args_file=args_file)
    return TransformerEncoderSentenceClassification(xlmr_model, sentence_classifier), xlmr_transform


TASK_PRETRAINED = {'xlmr_base_sentence_classifier': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_base_sentence_classifier-7e3fbb3f.pt'}
TASK_SHA256 = {'xlmr_base_sentence_classifier': '7e3fbb3fac705df2be377d9e1cc198ce3a578172a17b1943e94fa2efe592f278'}
