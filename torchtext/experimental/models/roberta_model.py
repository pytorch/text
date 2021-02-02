import os
import torch
import torch.nn as nn
from torchtext.experimental.modules import TransformerEncoder
from .roberta_transform import RobertaTransform
from torchtext.utils import download_from_url, extract_archive, files_exist, load_args_from_json


class RobertaModel(nn.Module):
    """Contain a Roberta encoder model."""

    def __init__(self, args, encoder):
        super(RobertaModel, self).__init__()
        self.args = args
        self.encoder = encoder
        self.transform = None

    @classmethod
    def build_model(self, args):
        encoder = TransformerEncoder(args.ntoken, args.embed_dim, args.nhead,
                                     args.feedforward_dim, args.nlayers, args.dropout)
        return cls(args, encoder)

    def load_transform(self, directory='./', tokenizer_file="sentencepiece.bpe.model", vocab_file='vocab.txt'):
        self.transform = RobertaTransform.from_pretrained(directory=directory,
                                                          tokenizer_file=tokenizer_file,
                                                          vocab_file=vocab_file)

    # [TODO] use torch.hub.load_state_dict_from_url
    @classmethod
    def from_pretrained(cls, args, directory='./', checkpoint_file="model.pt"):
        encoder = TransformerEncoder(args.ntoken, args.embed_dim, args.nhead,
                                     args.feedforward_dim, args.nlayers, args.dropout)
        filepath = os.path.join(directory, checkpoint_file)
        encoder.load_state_dict(torch.load(filepath))
        return cls(args, encoder)

    def forward(self, src):
        src = self.bert_embed(src)
        output = self.transformer_encoder(src)
        return output

    def encode(self, input_src: str) -> torch.Tensor:
        return torch.tensor(self.transform(input_src), dtype=torch.long)


# [TODO] Add torch.hub support
# [TODO] Download file from manifold
# [TODO] check base model config
# [TODO] Add xlmr large model
# [TODO] Remove file names. Those are not necessary
def xlmr_base_model(directory='./', checkpoint_file='model.pt', args_file='args.json',
                    tokenizer_file='sentencepiece.bpe.model', vocab_file='vocab.txt'):
    '''

    Examples:
        >>> from torchtext.experimental.models import xlmr_base_model
        >>> xlmr_base_model = xlmr_base_model()
        >>> xlmr_base_model.encode('this is an example')
        >>> tensor([  903,    83,   142, 27781])
    '''
    if not files_exist([checkpoint_file, args_file, tokenizer_file, vocab_file], root=directory):
        tar_file = download_from_url(PRETRAINED['xlmr.base'], root=directory,
                                     hash_value=MD5['xlmr.base'], hash_type='md5')
        extracted_files = extract_archive(tar_file, to_path=directory)

    args = load_args_from_json(args_file, root=directory)
    pretrained_model = RobertaModel.from_pretrained(args, directory=directory,
                                                    checkpoint_file=checkpoint_file)
    pretrained_model.load_transform(directory=directory,
                                    tokenizer_file=tokenizer_file,
                                    vocab_file=vocab_file)
    return pretrained_model


PRETRAINED = {'xlmr.base': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr.base.tar.gz'}
MD5 = {'xlmr.base': 'adf75f3d20c8a876533b206ccb3a7cb6'}

##################################################################################
# This part will be moved to stl-text/models folder

# [TODO] Add RobertaDocClassificationModel class
# [TODO] The RobertaDocClassificationModel model is composed of roberta encoder (from torchtext), classification head
# [TODO] def xlmr_doc_classification() func builds roberta encoder + classification head \
# and pass to the RobertaDocClassificationModel constructor.
