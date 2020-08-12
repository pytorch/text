import time

import torch
from torchtext.experimental.datasets.raw import AG_NEWS
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.data.utils import get_tokenizer


def benchmark_basic_english_normalize():
    def _run_benchmark_lookup(train, tokenizer):
        t0 = time.monotonic()
        for (label, text) in train:
            tokenizer(text)
        print("Lookup time:", time.monotonic() - t0)

    existing_basic_english_tokenizer = get_tokenizer("basic_english")
    experimental_basic_english_normalize = basic_english_normalize()
    experimental_jit_basic_english_normalize = torch.jit.script(experimental_basic_english_normalize.to_ivalue())

    # existing eager lookup
    train, _ = AG_NEWS()
    print("BasicEnglishNormalize - Eager Mode")
    _run_benchmark_lookup(train, existing_basic_english_tokenizer)

    # experimental eager lookup
    train, _ = AG_NEWS()
    print("BasicEnglishNormalize Experimental - Eager Mode")
    _run_benchmark_lookup(train, experimental_basic_english_normalize)

    # experimental jit lookup
    train, _ = AG_NEWS()
    print("BasicEnglishNormalize Experimental - Jit Mode")
    _run_benchmark_lookup(train, experimental_jit_basic_english_normalize)


if __name__ == "__main__":
    benchmark_basic_english_normalize()
