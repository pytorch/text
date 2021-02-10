import torch.nn as nn
from typing import List


class XLMRTransform(nn.Module):
    """XLM-R encode transform."""

    def __init__(self, tokenizer, vocab):
        super(XLMRTransform, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def forward(self, input_src: str) -> List[int]:
        return self.vocab(self.tokenizer(input_src))
