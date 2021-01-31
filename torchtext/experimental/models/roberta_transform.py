import os
import torch
import torch.nn as nn
from torchtext.experimental.transforms import sentencepiece_tokenizer
from torchtext.experimental.vocab import load_vocab_from_file
from typing import List


class RobertaTransform(nn.Module):
    """Roberta encode transform."""

    def __init__(self, tokenizer, vocab):
        super(RobertaTransform, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab

    @classmethod
    def from_pretrained(cls, directory='./',
                        tokenizer_file="sentencepiece.bpe.model",
                        vocab_file='vocab.txt'):
        filepath = os.path.join(directory, tokenizer_file)
        tokenizer = sentencepiece_tokenizer(filepath)
        filepath = os.path.join(directory, vocab_file)
        with open(filepath, 'r') as f:
            vocab = load_vocab_from_file(f)
        return cls(tokenizer, vocab)

    def forward(self, input_src: str) -> List[int]:
        return self.vocab(self.tokenizer(input_src))
