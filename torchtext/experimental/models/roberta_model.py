import os
from typing import List
import argparse
import torch
import torch.nn as nn
from torchtext.experimental.modules import TransformerEncoder
from .roberta_transform import RobertaTransform
from torchtext.utils import download_from_url, extract_archive, files_exist


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

    def encode(self, input_src: str) -> List[int]:
        return self.transform(input_src)


# [TODO] Add JSON file to save args
# [TODO] Add torch.hub support
# [TODO] Download file from manifold
def xlmr_base_model(directory='./', checkpoint_file="model.pt",
                    tokenizer_file="sentencepiece.bpe.model", vocab_file='vocab.txt'):
    '''

    Examples:
        >>> from torchtext.experimental.models import xlmr_base_model
        >>> xlmr_base_model = xlmr_base_model()
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--ntoken', type=int, default=250002,
                        help='the vocab size')
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='size of word embeddings')
    parser.add_argument('--nhead', type=int, default=16,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--feedforward_dim', type=int, default=4096,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    args = parser.parse_args()

    if files_exist([checkpoint_file, tokenizer_file, vocab_file], root=directory):
        tar_file = download_from_url(PRETRAINED['xlmr.base'], root=directory,
                                     hash_value=MD5['xlmr.base'], hash_type='md5')
        extracted_files = extract_archive(tar_file, to_path=directory)

    pretrained_model = RobertaModel.from_pretrained(args, directory=directory,
                                                    checkpoint_file=checkpoint_file)
    pretrained_model.load_transform(directory=directory,
                                    tokenizer_file=tokenizer_file,
                                    vocab_file=vocab_file)
    return pretrained_model


PRETRAINED = {'xlmr.base': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr.base.tar.gz'}
MD5 = {'xlmr.base': 'c33ae5b2377624dc077b659bbad222f3'}
