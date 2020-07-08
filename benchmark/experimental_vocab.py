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
        print("Lookup time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    for (_label, text) in train:
        for id in text.tolist():
            tokens.append(vocab.itos[id])

    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # existing Vocab construction
    print("Vocab")
    t0 = time.monotonic()
    v_existing = Vocab(counter)
    print("Construction time:", time.monotonic() - t0)

    # experimental Vocab construction
    print("Vocab Experimental")
    t0 = time.monotonic()
    v_experimental = VocabExperimental(ordered_dict)
    print("Construction time:", time.monotonic() - t0)
    jit_v_experimental = torch.jit.script(v_experimental)

    # existing Vocab not jit lookup
    print("Vocab - Not Jit Mode")
    _run_benchmark_lookup(tokens, v_existing)

    # experimental Vocab not jit lookup
    print("Vocab Experimental - Not Jit Mode")
    _run_benchmark_lookup(tokens, v_experimental)

    # experimental Vocab jit lookup
    print("Vocab Experimental - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v_experimental)


if __name__ == "__main__":
    benchmark_experimental_vocab()
