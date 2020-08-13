from collections import Counter, OrderedDict
from typing import List

import torch
import torch.nn as nn

from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.transforms import (
    BasicEnglishNormalize,
    TextSequentialTransforms,
)
from torchtext.experimental.vocab import Vocab as VocabExperimental


class VocabTransform(nn.Module):
    r"""Vocab transform
    """

    def __init__(self, vocab):
        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)


class TestTokenizer(nn.Module):
    def __init__(self):
        super(TestTokenizer, self).__init__()

    def forward(self, line: str) -> List[str]:
        return line.split()


def create_and_save_cpp_pipeline():
    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    for (_, text) in train:
        for id in text.tolist():
            tokens.append(vocab.itos[id])

    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    v_experimental = VocabExperimental(ordered_dict)

    # experimental_basic_english_normalize = BasicEnglishNormalize()

    test_tokenizer = TestTokenizer()
    # pipeline = TextSequentialTransforms(TestTokenizer
    pipeline = TextSequentialTransforms(test_tokenizer, VocabTransform(v_experimental))
    # pipeline = TextSequentialTransforms(experimental_basic_english_normalize, VocabTransform(v_experimental))
    jit_pipeline = torch.jit.script(pipeline)

    jit_pipeline.save("jit_pipeline.pt")

    # print(jit_pipeline("test string random"))

    print("done")


if __name__ == "__main__":
    create_and_save_cpp_pipeline()
