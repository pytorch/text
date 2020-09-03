import argparse
from collections import (Counter, OrderedDict)
import time

import torch
from torchtext.experimental.datasets import AG_NEWS
from torchtext.experimental.vocab import (
    vocab as VocabExperimental,
    vocab_from_file,
    vocab_from_raw_text_file
)
from torchtext.vocab import (
    Vocab,
    build_vocab_from_iterator
)
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.vocab import build_vocab_from_iterator

def _infer_shape(f):
    num_lines = 0
    for line in f:
        num_lines += 1
    f.seek(0)
    return num_lines


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
        >>> from torchtext.experimental.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> v = vocab_from_file_object(f, specials=('<unk>', '<pad>', '<eos>'), specials_first=False)
    """
    from tqdm import tqdm

    tokenizer = basic_english_normalize()

    def tokenize(line):
        return tokenizer(line)

    def token_iterator(lines):
        for line in lines:
            for token in tokenize(line):
                yield token

    return build_vocab_from_iterator(token_iterator(file_like_object))


def benchmark_experimental_vocab_construction(vocab_file_path, is_raw_text=True, is_legacy=True, num_iters=1):
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
                jited_tokenizer = torch.jit.script(tokenizer.to_ivalue())
                vocab_from_raw_text_file(f, jited_tokenizer, num_cpus=20)
            print("Construction time:", time.monotonic() - t0)
    else:
        for _ in range(num_iters):
            vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)


def benchmark_experimental_vocab_lookup(vocab_file_path=None):
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

    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    for (_, text) in train:
        cur_tokens = []
        for id in text.tolist():
            cur_tokens.append(vocab.itos[id])
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
        v_existing = build_vocab_from_iterator(token_iterator(vocab_file_path))
        print("Construction time:", time.monotonic() - t0)

        # experimental Vocab construction
        print("Vocab Experimental")
        t0 = time.monotonic()
        f = open(vocab_file_path, 'r')
        v_experimental = vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)
    else:
        print("Loading Vocab from AG News")
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
    jit_v_experimental = torch.jit.script(v_experimental.to_ivalue())

    # existing Vocab eager lookup
    print("Vocab - Eager Mode")
    _run_benchmark_lookup(tokens, v_existing)
    _run_benchmark_lookup([tokens], v_existing)
    _run_benchmark_lookup(tokens_lists, v_existing)

    # experimental Vocab eager lookup
    print("Vocab Experimental - Eager Mode")
    _run_benchmark_lookup(tokens, v_experimental)
    _run_benchmark_lookup([tokens], v_experimental)
    _run_benchmark_lookup(tokens_lists, v_experimental)

    jit_v_experimental = torch.jit.script(v_experimental.to_ivalue())
    # experimental Vocab jit lookup
    print("Vocab Experimental - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v_experimental)
    _run_benchmark_lookup([tokens], jit_v_experimental)
    _run_benchmark_lookup(tokens_lists, jit_v_experimental)


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
    args = parser.parse_args()

    if args.run_construction_benchmark:
        print("is_legacy", args.is_legacy)
        benchmark_experimental_vocab_construction(args.vocab_filename_construction,
                                                  is_raw_text=args.is_raw_text, is_legacy=args.is_legacy)
    else:
        benchmark_experimental_vocab_lookup(args.vocab_filename_lookup)
