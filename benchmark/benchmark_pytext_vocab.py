import os
import sys
from collections import (Counter, OrderedDict)
import time
from typing import List

# this is needed because we want to add 'torchtext/examples/data_pipeline' directory to the
# `sys.path` variable in order to import the pytext_vocab (since its not a module)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data_pipeline"))

from pytext_vocab import ScriptVocab as ExperimentalScriptVocabulary
from pytext.torchscript.vocab import ScriptVocabulary as PytextScriptVocabulary
from pytext.data.utils import Vocabulary as PytextVocabulary
import torch
from torchtext.experimental.datasets import AG_NEWS


def _run_benchmark_lookup(tokens, vocab, num_iters=1):
    def _run_benchmark_pytext_vocab(toks: List[str], v: PytextVocabulary):
        for token in toks:
            v.lookup_all(token)

    def _run_benchmark_pytext_script_vocab(toks: List[str], v: PytextScriptVocabulary):
        for token in toks:
            v.lookup_indices_1d([token])

    def _run_benchmark_experimental_script_vocab(toks: List[str], v: ExperimentalScriptVocabulary):
        for token in toks:
            v[token]

    t0 = time.monotonic()
    if type(vocab) is PytextVocabulary:
        for _ in range(num_iters):
            _run_benchmark_pytext_vocab(tokens, vocab)
    elif type(vocab) is PytextScriptVocabulary:
        for _ in range(num_iters):
            _run_benchmark_pytext_script_vocab(tokens, vocab)
    else:
        for _ in range(num_iters):
            _run_benchmark_experimental_script_vocab(tokens, vocab)
    print("Lookup time:", time.monotonic() - t0)


def _run_benchmark_lookup_list_tokens(tokens_lists, vocab, num_iters=1):
    def _run_benchmark_pytext_vocab(tok_lists, v: PytextVocabulary):
        for cur_tokens in tok_lists:
            v.lookup_all(cur_tokens)

    def _run_benchmark_script_vocab(tok_lists, v):
        for cur_tokens in tok_lists:
            v.lookup_indices_1d(cur_tokens)

    t0 = time.monotonic()
    if type(vocab) is PytextVocabulary:
        for _ in range(num_iters):
            _run_benchmark_pytext_vocab(tokens_lists, vocab)
    else:
        for _ in range(num_iters):
            _run_benchmark_script_vocab(tokens_lists, vocab)
    print("Lookup time:", time.monotonic() - t0)


def _run_benchmark_lookup_jit_for_loop(tokens, vocab, num_iters=1):
    @torch.jit.script
    def _run_benchmark_pytext_script_vocab(toks: List[str], v: PytextScriptVocabulary):
        for token in toks:
            v.lookup_indices_1d([token])

    @torch.jit.script
    def _run_benchmark_experimental_script_vocab(toks: List[str], v: ExperimentalScriptVocabulary):
        for token in toks:
            v[token]

    t0 = time.monotonic()
    if type(vocab) is PytextScriptVocabulary:
        for _ in range(num_iters):
            _run_benchmark_pytext_script_vocab(tokens, vocab)
    else:
        for _ in range(num_iters):
            _run_benchmark_experimental_script_vocab(tokens, vocab)
    print("Lookup time:", time.monotonic() - t0)


def _run_benchmark_lookup_list_tokens_jit_for_loop(tokens_lists, vocab, num_iters=1):
    @torch.jit.script
    def _run_benchmark_pytext_script_vocab(tok_lists: List[List[str]], v: PytextScriptVocabulary):
        for cur_tokens in tok_lists:
            v.lookup_indices_1d(cur_tokens)

    @torch.jit.script
    def _run_benchmark_experimental_script_vocab(tok_lists: List[List[str]], v: ExperimentalScriptVocabulary):
        for cur_tokens in tok_lists:
            v.lookup_indices_1d(cur_tokens)

    t0 = time.monotonic()

    if type(vocab) is PytextScriptVocabulary:
        for _ in range(num_iters):
            _run_benchmark_pytext_script_vocab(tokens_lists, vocab)
    else:
        for _ in range(num_iters):
            _run_benchmark_experimental_script_vocab(tokens_lists, vocab)
    print("Lookup time:", time.monotonic() - t0)


def benchmark_experimental_vocab():
    train, = AG_NEWS(data_select='train')
    vocab = train.get_vocab()
    tokens = []
    tokens_lists = []

    for (_, text) in train:
        cur_tokens = []
        for id in text.tolist():
            cur_tokens.append(vocab.itos[id])
        tokens_lists.append(cur_tokens)
        tokens += cur_tokens
        if len(tokens) > 100:
            break

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
    jit_experimental_script_vocab = torch.jit.script(experimental_script_vocab)
    print("Construction time:", time.monotonic() - t0)

    # pytext Vocab eager lookup
    print("Pytext Vocabulary - Eager Mode")
    _run_benchmark_lookup(tokens, pytext_vocab)
    _run_benchmark_lookup_list_tokens([tokens], pytext_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, pytext_vocab)

    # pytext ScriptVocab eager lookup
    print("Pytext ScriptVocab - Eager Mode")
    _run_benchmark_lookup(tokens, pytext_script_vocab)
    _run_benchmark_lookup_list_tokens([tokens], pytext_script_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, pytext_script_vocab)

    # pytext ScriptVocab jit lookup
    print("Pytext ScriptVocab - Jit Mode")
    _run_benchmark_lookup(tokens, jit_pytext_script_vocab)
    _run_benchmark_lookup_list_tokens([tokens], jit_pytext_script_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, jit_pytext_script_vocab)

    # experimental ScriptVocab eager lookup
    print("Experimental ScriptVocab - Eager Mode")
    _run_benchmark_lookup(tokens, experimental_script_vocab)
    _run_benchmark_lookup_list_tokens([tokens], experimental_script_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, experimental_script_vocab)

    # experimental ScriptVocab jit lookup
    print("Experimental ScriptVocab - Jit Mode")
    _run_benchmark_lookup(tokens, jit_experimental_script_vocab)
    _run_benchmark_lookup_list_tokens([tokens], jit_experimental_script_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, jit_experimental_script_vocab)

    # pytext ScriptVocab JITed for loop
    print("Pytext ScriptVocab - Jit For Loop")
    _run_benchmark_lookup_jit_for_loop(tokens, jit_pytext_script_vocab)
    _run_benchmark_lookup_list_tokens_jit_for_loop([tokens], jit_pytext_script_vocab)
    _run_benchmark_lookup_list_tokens_jit_for_loop(tokens_lists, jit_pytext_script_vocab)

    # experimental ScriptVocab JITed for loop
    print("Experimental ScriptVocab - Jit For Loop")
    _run_benchmark_lookup_jit_for_loop(tokens, jit_experimental_script_vocab)
    _run_benchmark_lookup_list_tokens_jit_for_loop([tokens], jit_experimental_script_vocab)
    _run_benchmark_lookup_list_tokens_jit_for_loop(tokens_lists, jit_experimental_script_vocab)


if __name__ == "__main__":
    benchmark_experimental_vocab()
