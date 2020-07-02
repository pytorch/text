from collections import (Counter, OrderedDict)
import time

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.vocab import Vocab as VocabExperimental
from torchtext.vocab import Vocab


def benchmark_experimental_vocab():
    def _run_benchmark_lookup(tokens, vocab):
        t0 = time.monotonic()
        for token in tokens:
            vocab[token]
        print("Time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    for (label, text) in train:
        for id in text.tolist():
            tokens.append(vocab.itos[id])
        if len(tokens) > 100:
            break

    counter = Counter(tokens)
    print(counter)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(sorted_by_freq_tuples)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # existing Vocab
    v = Vocab(counter)

    print("Vocab - Not Jit Mode")
    _run_benchmark_lookup(tokens, v)

    # experimental Vocab
    v = VocabExperimental(ordered_dict)
    jit_v = torch.jit.script(v)

    print("Vocab Experimental - Not Jit Mode")
    _run_benchmark_lookup(tokens, v)
    print("Vocab Experimental - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v)

if __name__ == "__main__":
    benchmark_experimental_vocab()
