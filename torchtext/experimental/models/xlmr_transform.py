import torch.nn as nn
from typing import List
from torchtext.experimental.transforms import sentencepiece_tokenizer
from torchtext.experimental.vocab import load_vocab_from_file


class XLMRTransform(nn.Module):
    """XLM-R encode transform."""

    def __init__(self, tokenizer, vocab):
        super(XLMRTransform, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def forward(self, input_src: str) -> List[int]:
        return self.vocab(self.tokenizer(input_src))


def load_xlmr_transform(tokenizer_file='sentencepiece.bpe.model', vocab_file='vocab.txt'):

    tokenizer = sentencepiece_tokenizer(tokenizer_file)
    with open(vocab_file, 'r') as f:
        vocab = load_vocab_from_file(f)
    return XLMRTransform(tokenizer, vocab)


TRANSFORM_PRETRAINED = {'xlmr_vocab': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_vocab-50081a8a.pt',
                        'xlmr_sentencepiece': 'https://pytorch.s3.amazonaws.com/models/text/pretrained_models/xlmr_sentencepiece-d4797664.pt'}
TRANSFORM_SHA256 = {'xlmr_vocab': '50081a8a69175ba2ed207eaf74f7055100aef3d8e737b3f0b26ee4a7c8fc781c',
                    'xlmr_sentencepiece': 'd47976646f6be0ae29b0f5d5b8a7b1d6381e46694a54e8d29dd51671f1471b33'}
