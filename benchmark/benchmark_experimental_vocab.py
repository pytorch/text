import argparse
from collections import (Counter, OrderedDict)
import time

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.vocab import (
    vocab as VocabExperimental,
    vocab_from_file_object
)
from torchtext.vocab import Vocab


def benchmark_experimental_vocab_construction(vocab_file_path, num_iters=100):
    f = open(vocab_file_path, 'r')
    t0 = time.monotonic()
    for i in range(num_iters):
        vocab_from_file_object(f)
    print("Construction time:", time.monotonic() - t0)


def benchmark_experimental_vocab_lookup():
    def _run_benchmark_lookup(tokens, vocab):
        t0 = time.monotonic()
        for token in tokens:
            vocab[token]
        print("Lookup time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    for (label, text) in train:
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

    # existing Vocab eager lookup
    print("Vocab - Eager Mode")
    _run_benchmark_lookup(tokens, v_existing)

    # experimental Vocab eager lookup
    print("Vocab Experimental - Eager Mode")
    _run_benchmark_lookup(tokens, v_experimental)

    jit_v_experimental = torch.jit.script(v_experimental.to_ivalue())
    # experimental Vocab jit lookup
    print("Vocab Experimental - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v_experimental)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data procesing pipelines')
    parser.add_argument('--run-construction-benchmark', type=bool, default=False,
                        help='run benchmark for constructing a vocab (default=False)')
    parser.add_argument('--vocab-filename', type=str, default='vocab.txt',
                        help='The name of vocab file')
    args = parser.parse_args()

    if args.run_construction_benchmark:
        benchmark_experimental_vocab_construction(args.vocab_filename)
    else:
        benchmark_experimental_vocab_lookup()
