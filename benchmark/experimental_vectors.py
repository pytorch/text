import time

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.vectors import FastText as FastTextExperimental
from torchtext.vocab import FastText


def benchmark_experimental_vectors():
    def _run_benchmark(tokens, vector):
        t0 = time.monotonic()
        for token in tokens:
            vector[token]
        print("Time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    for (label, text) in train:
        for id in text.tolist():
            tokens.append(vocab.itos[id])

    # existing FastText
    fast_text = FastText()

    print("FastText - Not Jit Mode")
    _run_benchmark(tokens, fast_text)

    # experimental FastText
    fast_text_experimental = FastTextExperimental()
    jit_fast_text_experimental = torch.jit.script(fast_text_experimental)

    print("FastText Experimental - Not Jit Mode")
    _run_benchmark(tokens, fast_text_experimental)
    print("FastText Experimental - Jit Mode")
    _run_benchmark(tokens, jit_fast_text_experimental)


if __name__ == "__main__":
    benchmark_experimental_vectors()
