import argparse
import time
from collections import Counter, OrderedDict

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import DATASETS
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.experimental.vocab_factory import build_vocab_from_text_file, load_vocab_from_file
from torchtext.vocab import build_vocab_from_iterator, vocab as VocabNew


def build_vocab(data, transforms):
    def apply_transforms(data):
        for _, line in data:
            yield transforms(line)

    vocab = build_vocab_from_iterator(apply_transforms(data), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def benchmark_new_vocab_construction(vocab_file_path, is_raw_text=True, num_iters=1):
    f = open(vocab_file_path, "r")
    t0 = time.monotonic()
    if is_raw_text:
        print("Loading from raw text file with basic_english_normalize tokenizer")
        for _ in range(num_iters):
            tokenizer = basic_english_normalize()
            jited_tokenizer = torch.jit.script(tokenizer)
            build_vocab_from_text_file(vocab_file_path, jited_tokenizer, num_cpus=1)
        print("Construction time:", time.monotonic() - t0)
    else:
        for _ in range(num_iters):
            load_vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)


def benchmark_new_vocab_lookup(vocab_file_path=None, dataset="AG_NEWS"):
    def _run_benchmark_lookup(tokens, vocab):
        t0 = time.monotonic()
        # list lookup
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            for tokens_list in tokens:
                vocab.lookup_indices(tokens_list)
        # single token lookup
        elif isinstance(tokens, list):
            for token in tokens:
                vocab[token]
        else:
            raise RuntimeError("Received tokens of incorrect type {}.".format(type(tokens)))
        print("Lookup time:", time.monotonic() - t0)

    tokens = []
    tokens_lists = []
    tokenizer = get_tokenizer("basic_english")
    for (_, text) in DATASETS[dataset](split="train"):
        cur_tokens = tokenizer(text)
        tokens_lists.append(cur_tokens)
        tokens += cur_tokens

    if vocab_file_path:
        print("Loading Vocab from file {}".format(vocab_file_path))

        def token_iterator(file_path):
            f = open(file_path, "r")
            for token in f:
                yield token

        # new Vocab construction
        print("Vocab New")
        t0 = time.monotonic()
        f = open(vocab_file_path, "r")
        v_new = load_vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)
    else:
        print("Loading Vocab from {}".format(dataset))
        counter = Counter(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        # new Vocab construction
        print("Vocab New")
        t0 = time.monotonic()
        v_new = VocabNew(ordered_dict)
        print("Construction time:", time.monotonic() - t0)
    jit_v_new = torch.jit.script(v_new)

    # new Vocab eager lookup
    print("Vocab New - Eager Mode")
    _run_benchmark_lookup(tokens, v_new)
    _run_benchmark_lookup([tokens], v_new)
    _run_benchmark_lookup(tokens_lists, v_new)

    jit_v_new = torch.jit.script(v_new)
    # new Vocab jit lookup
    print("Vocab New - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v_new)
    _run_benchmark_lookup([tokens], jit_v_new)
    _run_benchmark_lookup(tokens_lists, jit_v_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data procesing pipelines")
    parser.add_argument(
        "--run-construction-benchmark",
        type=bool,
        default=False,
        help="run benchmark for constructing a vocab (default=False)",
    )
    parser.add_argument(
        "--is-raw-text", type=bool, default=True, help="construct vocab from raw text file (default=True)"
    )
    parser.add_argument(
        "--vocab-filename-construction",
        type=str,
        default="vocab.txt",
        help="The name of vocab file used for construction",
    )
    parser.add_argument(
        "--vocab-filename-lookup", type=str, default=None, help="The name of vocab file used for lookup"
    )
    parser.add_argument("--dataset", type=str, default="AG_NEWS", help="The name of vocab file used for lookup")
    args = parser.parse_args()

    if args.run_construction_benchmark:
        benchmark_new_vocab_construction(args.vocab_filename_construction, is_raw_text=args.is_raw_text)
    else:
        benchmark_new_vocab_lookup(args.vocab_filename_lookup, args.dataset)
