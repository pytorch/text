import torch
from torchtext.experimental.modules import TransformerEncoder
from .xlmr_transform import XLMRTransform
from torchtext.utils import download_from_url, extract_archive, load_args_from_json
from torchtext.experimental.transforms import sentencepiece_tokenizer
from torchtext.experimental.vocab import load_vocab_from_file


# [TODO] Add torch.hub support
# [TODO] Download file from manifold
# [TODO] check base model config
def xlmr_base(root='./.model'):
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_base
        >>> xlmr_base_model, xlmr_base_transform = xlmr_base()
        >>> xlmr_base_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    tar_file = download_from_url(PRETRAINED['xlmr.base'], root=root,
                                 hash_value=MD5['xlmr.base'], hash_type='md5')
    checkpoint_file, tokenizer_file, vocab_file, args_file = extract_archive(tar_file)
    return _load_xlmr_model(checkpoint_file=checkpoint_file, args_file=args_file), _load_xlmr_transform(tokenizer_file=tokenizer_file, vocab_file=vocab_file)


def xlmr_regular(root='./.model'):
    '''
    Examples:
        >>> from torchtext.experimental.models import xlmr_regular
        >>> xlmr_regular_model, xlmr_regular_transform = xlmr_regular()
        >>> xlmr_regular_transform('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    tar_file = download_from_url(PRETRAINED['xlmr.regular'], root=root,
                                 hash_value=MD5['xlmr.regular'], hash_type='md5')
    checkpoint_file, tokenizer_file, vocab_file, args_file = extract_archive(tar_file)
    return _load_xlmr_model(checkpoint_file=checkpoint_file, args_file=args_file), _load_xlmr_transform(tokenizer_file=tokenizer_file, vocab_file=vocab_file)


def _load_xlmr_model(checkpoint_file='model.pt', args_file='args.json'):
    args = load_args_from_json(args_file)
    encoder = TransformerEncoder(args.ntoken, args.embed_dim, args.nhead,
                                 args.feedforward_dim, args.nlayers, args.dropout)
    encoder.load_state_dict(torch.load(checkpoint_file))
    return encoder


def _load_xlmr_transform(tokenizer_file='sentencepiece.bpe.model', vocab_file='vocab.txt'):

    tokenizer = sentencepiece_tokenizer(tokenizer_file)
    with open(vocab_file, 'r') as f:
        vocab = load_vocab_from_file(f)
    return XLMRTransform(tokenizer, vocab)


PRETRAINED = {'xlmr.regular': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr.regular.tar.gz',
              'xlmr.base': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr.base.tar.gz'}
MD5 = {'xlmr.regular': 'adf75f3d20c8a876533b206ccb3a7cb6',
       'xlmr.base': 'abc0a28171f883c6e1b2e8cf184c1ce8'}

##################################################################################
# This part will be moved to stl-text/models folder

# [TODO] Add RobertaDocClassificationModel class
# [TODO] The RobertaDocClassificationModel model is composed of roberta encoder (from torchtext), classification head
# [TODO] def xlmr_doc_classification() func builds roberta encoder + classification head \
# and pass to the RobertaDocClassificationModel constructor.
