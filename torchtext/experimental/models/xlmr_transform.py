import os
import torch.nn as nn
from torchtext.experimental.transforms import sentencepiece_tokenizer
from torchtext.experimental.vocab import load_vocab_from_file
from typing import List


class XLMRTransform(nn.Module):
    """XLM-R encode transform."""

    def __init__(self, tokenizer, vocab):
        super(XLMRTransform, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def forward(self, input_src: str) -> List[int]:
        return self.vocab(self.tokenizer(input_src))


def xlmr_transform(root='./', tokenizer_file="sentencepiece.bpe.model",
                   vocab_file='vocab.txt'):
    filepath = os.path.join(directory, tokenizer_file)
    tokenizer = sentencepiece_tokenizer(filepath)
    filepath = os.path.join(directory, vocab_file)
    with open(filepath, 'r') as f:
        vocab = load_vocab_from_file(f)
    return XLMRTransform(tokenizer, vocab)
