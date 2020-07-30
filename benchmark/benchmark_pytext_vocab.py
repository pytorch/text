import os
import sys
from collections import (Counter, OrderedDict)
import time
from typing import List, Union

# this is needed because we want to add 'torchtext/examples/data_pipeline' directory to the
# `sys.path` variable in order to import the pytext_vocab (since its not a module)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "vocab"))

from pytext_vocab import ScriptVocab as ExperimentalScriptVocabulary
from pytext.torchscript.vocab import ScriptVocabulary as PytextScriptVocabulary
from pytext.data.utils import Vocabulary as PytextVocabulary
import torch
from torchtext.experimental.datasets import AG_NEWS


def _run_benchmark_lookup(tokens, vocab, num_iters=1):
    def _run_benchmark_pytext_vocab(toks, v: PytextVocabulary):
        for token_or_tokens_list in toks:
            v.lookup_all(token_or_tokens_list)

    def _run_benchmark_pytext_script_vocab(toks, v: PytextScriptVocabulary):
        # list lookup
        if isinstance(toks, list) and isinstance(toks[0], list):
            for tokens_list in toks:
                v.lookup_indices_1d(tokens_list)
        # single token lookup
        elif isinstance(toks, list):
            for token in toks:
                v.lookup_indices_1d([token])
        else:
            raise RuntimeError("Received tokens of incorrect type {}.".format(type(toks)))

    def _run_benchmark_experimental_script_vocab(toks, v: ExperimentalScriptVocabulary):
        # list lookup
        if isinstance(toks, list) and isinstance(toks[0], list):
            for tokens_list in toks:
                v.lookup_indices_1d(tokens_list)
        # single token lookup
        elif isinstance(toks, list):
            for token in toks:
                v[token]
        else:
            raise RuntimeError("Received tokens of incorrect type {}.".format(type(toks)))

    t0 = time.monotonic()
    if isinstance(vocab, PytextVocabulary):
        for _ in range(num_iters):
            _run_benchmark_pytext_vocab(tokens, vocab)
    elif isinstance(vocab, PytextScriptVocabulary):
        for _ in range(num_iters):
            _run_benchmark_pytext_script_vocab(tokens, vocab)
    elif isinstance(vocab, (ExperimentalScriptVocabulary, torch.jit._script.RecursiveScriptModule)):
        for _ in range(num_iters):
            _run_benchmark_experimental_script_vocab(tokens, vocab)
    else:
        raise RuntimeError("Received vocab of incorrect type {}.".format(type(vocab)))

    print("Lookup time:", time.monotonic() - t0)


def _run_benchmark_lookup_jit_for_loop(tokens: Union[List[str], List[List[str]]], vocab, num_iters=1):
    @torch.jit.script
    def _run_benchmark_pytext_script_vocab(toks: List[str], v: PytextScriptVocabulary):
        for token in toks:
            v.lookup_indices_1d([token])

    @torch.jit.script
    def _run_benchmark_experimental_script_vocab(toks: List[str], v: ExperimentalScriptVocabulary):
        for token in toks:
            v[token]

    @torch.jit.script
    def _run_benchmark_lists_pytext_script_vocab(tok_lists: List[List[str]], v: PytextScriptVocabulary):
        for tokens_list in tok_lists:
            v.lookup_indices_1d(tokens_list)

    @torch.jit.script
    def _run_benchmark_lists_experimental_script_vocab(tok_lists: List[List[str]], v: ExperimentalScriptVocabulary):
        for tokens_list in tok_lists:
            v.lookup_indices_1d(tokens_list)

    t0 = time.monotonic()
    # list lookup
    if isinstance(tokens, list) and isinstance(tokens[0], list):
        if isinstance(vocab, PytextScriptVocabulary):
            for _ in range(num_iters):
                _run_benchmark_lists_pytext_script_vocab(tokens, vocab)
        elif isinstance(vocab, (ExperimentalScriptVocabulary, torch.jit._script.RecursiveScriptModule)):

            for _ in range(num_iters):
                _run_benchmark_lists_experimental_script_vocab(tokens, vocab)
        else:
            raise RuntimeError("Received vocab of incorrect type {}.".format(type(vocab)))
    # single token lookup
    elif isinstance(tokens, list):
        if isinstance(vocab, PytextScriptVocabulary):
            for _ in range(num_iters):
                _run_benchmark_pytext_script_vocab(tokens, vocab)
        elif isinstance(vocab, (ExperimentalScriptVocabulary, torch.jit._script.RecursiveScriptModule)):
            for _ in range(num_iters):
                _run_benchmark_experimental_script_vocab(tokens, vocab)
        else:
            raise RuntimeError("Received vocab of incorrect type {}.".format(type(vocab)))
    else:
        raise RuntimeError("Received tokens of incorrect type {}.".format(type(tokens)))

    print("Lookup time:", time.monotonic() - t0)


def benchmark_experimental_vocab():
    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens: List[str] = []
    tokens_lists: List[List[str]] = []

    for (_, text) in train:
        cur_tokens = []
        for id in text.tolist():
            cur_tokens.append(vocab.itos[id])
        tokens_lists.append(cur_tokens)
        tokens += cur_tokens

    print("Tokens size:", len(tokens))
    print("Tokens list size:", len(tokens_lists))

    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    vocab_list = [pair[0] for pair in sorted_by_freq_tuples]
    vocab_list.insert(0, "<unk>")
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # pytext vocab construction
    print("Pytext Vocabulary")
    t0 = time.monotonic()
    pytext_vocab = PytextVocabulary(vocab_list)
    print("Construction time:", time.monotonic() - t0)

    # pytext ScriptVocab construction
    print("Pytext Script Vocabulary")
    t0 = time.monotonic()
    pytext_script_vocab = PytextScriptVocabulary(vocab_list)
    print("Construction time:", time.monotonic() - t0)
    jit_pytext_script_vocab = torch.jit.script(pytext_script_vocab)

    # experimental ScriptVocab construction
    print("Experimental Script Vocabulary")
    t0 = time.monotonic()
    experimental_script_vocab = ExperimentalScriptVocabulary(ordered_dict, unk_token="<unk>")
    print("Construction time:", time.monotonic() - t0)
    jit_experimental_script_vocab = torch.jit.script(experimental_script_vocab)

    # pytext Vocab eager lookup
    print("Pytext Vocabulary - Eager Mode")
    _run_benchmark_lookup(tokens, pytext_vocab)
    _run_benchmark_lookup([tokens], pytext_vocab)
    _run_benchmark_lookup(tokens_lists, pytext_vocab)

    # pytext ScriptVocab eager lookup
    print("Pytext ScriptVocab - Eager Mode")
    _run_benchmark_lookup(tokens, pytext_script_vocab)
    _run_benchmark_lookup([tokens], pytext_script_vocab)
    _run_benchmark_lookup(tokens_lists, pytext_script_vocab)

    # experimental ScriptVocab eager lookup
    print("Experimental ScriptVocab - Eager Mode")
    _run_benchmark_lookup(tokens, experimental_script_vocab)
    _run_benchmark_lookup([tokens], experimental_script_vocab)
    _run_benchmark_lookup(tokens_lists, experimental_script_vocab)

    # pytext ScriptVocab jit lookup
    print("Pytext ScriptVocab - Jit Mode")
    _run_benchmark_lookup(tokens, jit_pytext_script_vocab)
    _run_benchmark_lookup([tokens], jit_pytext_script_vocab)
    _run_benchmark_lookup(tokens_lists, jit_pytext_script_vocab)

    # experimental ScriptVocab jit lookup
    print("Experimental ScriptVocab - Jit Mode")
    _run_benchmark_lookup(tokens, jit_experimental_script_vocab)
    _run_benchmark_lookup([tokens], jit_experimental_script_vocab)
    _run_benchmark_lookup(tokens_lists, jit_experimental_script_vocab)

    # pytext ScriptVocab JITed for loop
    print("Pytext ScriptVocab - Jit For Loop")
    _run_benchmark_lookup_jit_for_loop(tokens, jit_pytext_script_vocab)
    _run_benchmark_lookup_jit_for_loop([tokens], jit_pytext_script_vocab)
    _run_benchmark_lookup_jit_for_loop(tokens_lists, jit_pytext_script_vocab)

    # experimental ScriptVocab JITed for loop
    print("Experimental ScriptVocab - Jit For Loop")
    _run_benchmark_lookup_jit_for_loop(tokens, jit_experimental_script_vocab)
    _run_benchmark_lookup_jit_for_loop([tokens], jit_experimental_script_vocab)
    _run_benchmark_lookup_jit_for_loop(tokens_lists, jit_experimental_script_vocab)


if __name__ == "__main__":
    benchmark_experimental_vocab()
