from collections import (Counter, OrderedDict)
import time

import torch
from torchtext.experimental.datasets import AG_NEWS

from experimental_vocab import ScriptVocab as ExperimentalScriptVocabulary
from pytext.torchscript.vocab import ScriptVocabulary


def benchmark_experimental_vocab():
    def _run_benchmark_lookup_list_tokens(tokens_lists, vocab, num_iters=1):
        t0 = time.monotonic()
        for _ in range(num_iters):
            for cur_tokens in tokens_lists:
                vocab.lookup_indices_1d(cur_tokens)
        print("Lookup time:", time.monotonic() - t0)

    def _run_benchmark_lookup(tokens, vocab, is_pytext_vocab=True, num_iters=1):
        t0 = time.monotonic()
        if is_pytext_vocab:
            for _ in range(num_iters):
                for token in tokens:
                    vocab.lookup_indices_1d([token])
        else:
            for _ in range(num_iters):
                for token in tokens:
                    vocab[token]
        print("Lookup time:", time.monotonic() - t0)

    train, = AG_NEWS(data_select='train')

    vocab = train.get_vocab()
    tokens = []
    tokens_lists = []

    for (label, text) in train:
        cur_tokens = []
        for id in text.tolist():
            cur_tokens.append(vocab.itos[id])
        tokens_lists.append(cur_tokens)
        tokens += cur_tokens

    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    vocab_list = [pair[0] for pair in sorted_by_freq_tuples]
    vocab_list.insert(0, "<unk>")
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    print(len(ordered_dict), len(vocab_list))

    # Script Vocab construction
    print("Script Vocabulary")
    t0 = time.monotonic()
    script_vocab = ScriptVocabulary(vocab_list)
    print("Construction time:", time.monotonic() - t0)
    jit_script_vocab = torch.jit.script(script_vocab)

    # experimental script Vocab construction
    print("Experimental Script Vocabulary")
    t0 = time.monotonic()
    script_vocab_experimental = ExperimentalScriptVocabulary(ordered_dict, unk_token="<unk>")
    jit_script_vocab_experimental = torch.jit.script(script_vocab_experimental)
    print("Construction time:", time.monotonic() - t0)

    # script Vocab not jit lookup
    print("Script Vocabulary - Not Jit Mode")
    # _run_benchmark_lookup(tokens, script_vocab)
    # _run_benchmark_lookup_list_tokens([tokens], script_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, script_vocab)

    # script Vocab jit lookup
    print("Script Vocabulary - Jit Mode")
    # _run_benchmark_lookup(tokens, jit_script_vocab)
    # _run_benchmark_lookup_list_tokens([tokens], jit_script_vocab)
    _run_benchmark_lookup_list_tokens(tokens_lists, jit_script_vocab)

    # experimental script Vocab not jit lookup
    print("Experimental Script Vocabulary - Not Jit Mode")
    # _run_benchmark_lookup(tokens, script_vocab_experimental, is_pytext_vocab=False)
    # _run_benchmark_lookup_list_tokens([tokens], script_vocab_experimental)
    _run_benchmark_lookup_list_tokens(tokens_lists, script_vocab_experimental)

    # experimental script Vocab jit lookup
    print("Experimental Script Vocabulary - Jit Mode")
    # _run_benchmark_lookup(tokens, jit_script_vocab_experimental, is_pytext_vocab=False)
    # _run_benchmark_lookup_list_tokens([tokens], jit_script_vocab_experimental)
    _run_benchmark_lookup_list_tokens(tokens_lists, jit_script_vocab_experimental)


if __name__ == "__main__":
    benchmark_experimental_vocab()
