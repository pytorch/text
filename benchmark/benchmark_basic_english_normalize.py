import time

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.transforms import BasicEnglishNormalize
from torchtext.data.utils import get_tokenizer


def benchmark_experimental_vocab():
    def _run_benchmark_lookup(train, tokenizer):
        t0 = time.monotonic()
        for (label, text) in train.data:
            tokenizer(text)
        print("Lookup time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')
    
    existing_basic_english_tokenizer = get_tokenizer("basic_english")
    experimental_basic_english_normalize = BasicEnglishNormalize()
    experimental_jit_basic_english_normalize = torch.jit.script(experimental_basic_english_normalize)

    # existing eager lookup
    print("Vocab - Eager Mode")
    _run_benchmark_lookup(train, existing_basic_english_tokenizer)

    # experimental eager lookup
    print("Vocab Experimental - Eager Mode")
    _run_benchmark_lookup(train, experimental_basic_english_normalize)

    # experimental jit lookup
    print("Vocab Experimental - Jit Mode")
    _run_benchmark_lookup(train, experimental_jit_basic_english_normalize)


if __name__ == "__main__":
    benchmark_experimental_vocab()
