from torchtext import vocab
from torchtext import experimental
from torchtext.data.utils import get_tokenizer
import time

def benchmark_fasttext():
    ft = vocab.FastText()
    train, test = experimental.datasets.raw.AG_NEWS()
    tokenizer = get_tokenizer('basic_english')
    strings = list(tokenizer(s) for (l, s) in train)
    t0 = time.monotonic()
    for sl in strings:
        ft.get_vecs_by_tokens(sl)
    print(time.monotonic() - t0)


if __name__ == "__main__":
    benchmark_fasttext()
