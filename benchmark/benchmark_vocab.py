import argparse
from collections import (Counter, OrderedDict)
import time
import random
import string
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import torch
from torchtext.datasets import DATASETS
from torchtext.experimental.vocab_factory import (
    load_vocab_from_file,
    build_vocab_from_text_file
)
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab as VocabNew
from torchtext.legacy.vocab import (
    Vocab,
    build_vocab_from_iterator as build_vocab_from_iterator_legacy,
)
from torchtext.experimental.transforms import(
    basic_english_normalize,
)
from torchtext.data.utils import get_tokenizer

def build_vocab(data, transforms):
    def apply_transforms(data):
        for _, line in data:
            yield transforms(line)
    vocab = build_vocab_from_iterator(apply_transforms(data), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def compare_legacy_and_new_batch_lookup():
    num_tokens = 1000
    num_letters = 6
    num_lines = 100000
    vocab = [''.join(random.sample(string.ascii_letters * num_letters, num_letters)) for _ in range(num_tokens)]
    counter = Counter()
    counter.update(vocab)
    legacy_vocab = Vocab(counter)
    new_vocab = VocabNew(counter)
    speed_ups = []
    token_lengths = [i for i in range(2, 100)]
    for i in token_lengths:
        lines = [random.sample(vocab, i) for _ in range(num_lines)]
        start_time = timer()
        for text in lines:
            legacy_vocab.lookup_indices(text)
        legacy_time = timer() - start_time

        start_time = timer()
        for text in lines:
            new_vocab.lookup_indices(text)

        new_time = timer() - start_time

        speed_ups.append(legacy_time / new_time)
        print("speed-up={} for average length={}".format(legacy_time / new_time, i))
        del lines

    plt.close()
    fig, ax = plt.subplots(1, 1)
    ax.plot(token_lengths, speed_ups)
    ax.set_xlabel('Average Tokens per line')
    ax.set_ylabel('Speed-up')
    plt.savefig("speedup.jpg")


def legacy_vocab_from_file_object(file_like_object, **kwargs):
    r"""Create a `Vocab` object from a file like object.

    The `file_like_object` should contain tokens seperated by new lines. Note that the vocab
    will be created in the order that the tokens first appear in the file (and not by the frequency of tokens).

    Format for txt file:
        token1
        token2
        ...
        token_n

    Args:
        file_like_object (FileObject): a file like object to read data from.
        Remaining keyword arguments: Passed to the constructor of Vocab class.

    Returns:
        Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> v = vocab_from_file_object(f, specials=('<unk>', '<pad>', '<eos>'), specials_first=False)
    """
    tokenizer = basic_english_normalize()

    def tokenize(line):
        return tokenizer(line)

    def token_iterator(lines):
        for line in lines:
            for token in tokenize(line):
                yield token

    return build_vocab_from_iterator_legacy(token_iterator(file_like_object))


def benchmark_new_vocab_construction(vocab_file_path, is_raw_text=True, is_legacy=True, num_iters=1):
    f = open(vocab_file_path, 'r')
    t0 = time.monotonic()
    if is_raw_text:
        if is_legacy:
            print("Loading from raw text file with legacy python function")
            for _ in range(num_iters):
                legacy_vocab_from_file_object(f)

            print("Construction time:", time.monotonic() - t0)
        else:
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


def benchmark_new_vocab_lookup(vocab_file_path=None, dataset='AG_NEWS'):
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
    for (_, text) in DATASETS[dataset](split='train'):
       cur_tokens = tokenizer(text)
       tokens_lists.append(cur_tokens)
       tokens += cur_tokens

    if vocab_file_path:
        print("Loading Vocab from file {}".format(vocab_file_path))

        def token_iterator(file_path):
            f = open(file_path, 'r')
            for token in f:
                yield token

        # existing Vocab construction
        print("Vocab")
        t0 = time.monotonic()
        v_existing = build_vocab_from_iterator_legacy(token_iterator(vocab_file_path))
        print("Construction time:", time.monotonic() - t0)

        # new Vocab construction
        print("Vocab New")
        t0 = time.monotonic()
        f = open(vocab_file_path, 'r')
        v_new = load_vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)
    else:
        print("Loading Vocab from {}".format(dataset))
        counter = Counter(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        # existing Vocab construction
        print("Vocab")
        t0 = time.monotonic()
        v_existing = Vocab(counter)
        print("Construction time:", time.monotonic() - t0)

        # new Vocab construction
        print("Vocab New")
        t0 = time.monotonic()
        v_new = VocabNew(ordered_dict)
        print("Construction time:", time.monotonic() - t0)
    jit_v_new = torch.jit.script(v_new)

    # existing Vocab eager lookup
    print("Vocab - Eager Mode")
    _run_benchmark_lookup(tokens, v_existing)
    _run_benchmark_lookup([tokens], v_existing)
    _run_benchmark_lookup(tokens_lists, v_existing)

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
    parser = argparse.ArgumentParser(description='Data procesing pipelines')
    parser.add_argument('--run-construction-benchmark', type=bool, default=False,
                        help='run benchmark for constructing a vocab (default=False)')
    parser.add_argument('--is-raw-text', type=bool, default=True,
                        help='construct vocab from raw text file (default=True)')
    parser.add_argument('--is-legacy', type=bool, default=False,
                        help='construct vocab using legacy implementation (default=False)')
    parser.add_argument('--vocab-filename-construction', type=str, default='vocab.txt',
                        help='The name of vocab file used for construction')
    parser.add_argument('--vocab-filename-lookup', type=str, default=None,
                        help='The name of vocab file used for lookup')
    parser.add_argument('--dataset', type=str, default='AG_NEWS',
                        help='The name of vocab file used for lookup')
    args = parser.parse_args()

    if args.run_construction_benchmark:
        print("is_legacy", args.is_legacy)
        benchmark_new_vocab_construction(args.vocab_filename_construction,
                                                  is_raw_text=args.is_raw_text, is_legacy=args.is_legacy)
    else:
        benchmark_new_vocab_lookup(args.vocab_filename_lookup, args.dataset)
