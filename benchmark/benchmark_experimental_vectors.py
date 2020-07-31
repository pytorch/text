import time

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.vectors import FastText as FastTextExperimental
from torchtext.vocab import FastText


def benchmark_experimental_vectors():
    def _run_benchmark_lookup(tokens, vector):
        t0 = time.monotonic()
        for token in tokens:
            vector[token]
        print("Lookup time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    for (label, text) in train:
        for id in text.tolist():
            tokens.append(vocab.itos[id])

    # existing FastText construction
    print("Existing FastText - Not Jit Mode")
    t0 = time.monotonic()
    fast_text = FastText()
    print("Construction time:", time.monotonic() - t0)
    _run_benchmark_lookup(tokens, fast_text)

    # experimental FastText construction
    print("FastText Experimental")
    t0 = time.monotonic()
    fast_text_experimental = FastTextExperimental(validate_file=False)
    print("Construction time:", time.monotonic() - t0)

    # not jit lookup
    print("FastText Experimental - Not Jit Mode")
    _run_benchmark_lookup(tokens, fast_text_experimental)

    # jit lookup
    print("FastText Experimental - Jit Mode")
    jit_fast_text_experimental = torch.jit.script(fast_text_experimental)
    _run_benchmark_lookup(tokens, jit_fast_text_experimental)


if __name__ == "__main__":
    benchmark_experimental_vectors()
