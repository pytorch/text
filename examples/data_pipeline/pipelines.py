import argparse
import time
from collections import Counter, OrderedDict

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import DATASETS
from torchtext.experimental.functional import sequential_transforms
from torchtext.experimental.transforms import (
    basic_english_normalize,
    load_sp_model,
    PRETRAINED_SP_MODEL,
    sentencepiece_tokenizer,
    TextSequentialTransforms,
)
from torchtext.experimental.vectors import FastText as FastTextExperimental
from torchtext.experimental.vocab import load_vocab_from_file
from torchtext.utils import download_from_url
from torchtext.vocab import FastText
from transforms import PretrainedSPVocab, PyTextScriptVocabTransform, PyTextVocabTransform, tokenizer_func, vocab_func


def build_sp_pipeline(args):
    spm_file = args.spm_filename
    if spm_file in PRETRAINED_SP_MODEL:
        spm_file = download_from_url(PRETRAINED_SP_MODEL[spm_file])
    tokenizer = sentencepiece_tokenizer(spm_file)
    vocab = PretrainedSPVocab(load_sp_model(spm_file))

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer, vocab)
    jit_pipeline = torch.jit.script(pipeline)
    print("jit sentencepiece pipeline success!")
    return pipeline, pipeline, jit_pipeline


def build_legacy_torchtext_vocab_pipeline(args):
    vocab_file = args.vocab_filename
    tokenizer = get_tokenizer("basic_english")
    from torchtext.legacy.vocab import build_vocab_from_iterator

    def token_iterator(vocab_file):
        f = open(vocab_file, "r")
        for line in f:
            for token in line:
                yield token

    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    pipeline = sequential_transforms(tokenizer, vocab_func(vocab))
    return pipeline, None, None


def build_experimental_torchtext_pipeline(args):
    vocab_file = args.vocab_filename
    tokenizer = basic_english_normalize()
    with open(vocab_file, "r") as f:
        vocab = load_vocab_from_file(f)
        pipeline = TextSequentialTransforms(tokenizer, vocab)
        jit_pipeline = torch.jit.script(pipeline)
        print("jit experimental torchtext pipeline success!")
        return pipeline, pipeline, jit_pipeline


def build_legacy_batch_torchtext_vocab_pipeline(args):
    vocab_file = args.vocab_filename
    tokenizer = get_tokenizer("basic_english")
    from torchtext.legacy.vocab import build_vocab_from_iterator

    def token_iterator(vocab_file):
        f = open(vocab_file, "r")
        for line in f:
            for token in line:
                yield token

    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    text_pipeline = sequential_transforms(tokenizer_func(tokenizer), vocab_func(vocab))
    return text_pipeline, None, None


def build_legacy_pytext_vocab_pipeline(args):
    vocab_file = args.vocab_filename
    from pytext.data.utils import Vocabulary

    tokenizer = get_tokenizer("basic_english")
    with open(vocab_file, "r") as f:
        vocab_counter = Counter([token for line in f for token in line.rstrip()])
        sorted_by_freq_tuples = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
        vocab_list = [pair[0] for pair in sorted_by_freq_tuples]
        vocab_list.insert(0, "<unk>")
        pipeline = sequential_transforms(
            tokenizer_func(tokenizer), PyTextVocabTransform(Vocabulary(vocab_list, unk_token="<unk>"))
        )
        return pipeline, None, None


def build_legacy_pytext_script_vocab_pipeline(args):
    vocab_file = args.vocab_filename
    from pytext.torchscript.vocab import ScriptVocabulary

    tokenizer = basic_english_normalize()
    with open(vocab_file, "r") as f:
        vocab_counter = Counter([token for line in f for token in line.rstrip()])
        sorted_by_freq_tuples = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
        vocab_list = [pair[0] for pair in sorted_by_freq_tuples]
        vocab_list.insert(0, "<unk>")
        pipeline = TextSequentialTransforms(tokenizer, PyTextScriptVocabTransform(ScriptVocabulary(vocab_list)))
        jit_pipeline = torch.jit.script(pipeline)
        print("jit legacy PyText pipeline success!")
        return pipeline, pipeline, jit_pipeline


def build_experimental_pytext_script_pipeline(args):
    vocab_file = args.vocab_filename
    import os
    import sys

    # this is needed because we want to add 'torchtext/examples/vocab' directory to the
    # `sys.path` variable in order to import the pytext_vocab (since its not a module)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vocab"))
    from pytext_vocab import script_vocab

    tokenizer = basic_english_normalize()
    f = open(vocab_file, "r")
    vocab_counter = Counter([token for line in f for token in line.rstrip()])
    ordered_dict = OrderedDict(sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True))

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer, PyTextScriptVocabTransform(script_vocab(ordered_dict)))
    jit_pipeline = torch.jit.script(pipeline)
    print("jit legacy PyText pipeline success!")
    return pipeline, pipeline, jit_pipeline


def build_legacy_fasttext_vector_pipeline(args):
    tokenizer = get_tokenizer("basic_english")
    vector = FastText()

    pipeline = sequential_transforms(tokenizer, vector.get_vecs_by_tokens)
    return pipeline, None, None


def build_experimental_fasttext_vector_pipeline(args):
    tokenizer = basic_english_normalize()
    vector = FastTextExperimental()

    pipeline = TextSequentialTransforms(tokenizer, vector)
    jit_pipeline = torch.jit.script(pipeline)

    print("jit legacy fasttext pipeline success!")
    return pipeline, pipeline, jit_pipeline


def run_benchmark_lookup(text_classification_dataset, pipeline):
    t0 = time.monotonic()
    for (label, text) in text_classification_dataset:
        text = pipeline(text)
    print("Lookup time:", time.monotonic() - t0)


def run_batch_benchmark_lookup(text_classification_dataset, pipeline):
    def collate_fn(data_batch):
        return [text for (label, text) in data_batch]

    dataloader = DataLoader(text_classification_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    t0 = time.monotonic()
    for lines in dataloader:
        lines = pipeline(lines)
    print("Lookup time:", time.monotonic() - t0)


def generate_dataset(args):
    train, test = DATASETS[args.dataset]()
    return [_data for _data in train], [_data for _data in test]


PIPELINES = {
    "sentencepiece": build_sp_pipeline,
    "experimental_torchtext": build_experimental_torchtext_pipeline,
    "legacy_torchtext": build_legacy_torchtext_vocab_pipeline,
    "experimental_fasttext": build_experimental_fasttext_vector_pipeline,
    "legacy_fasttext": build_legacy_fasttext_vector_pipeline,
    "experimental_pytext_script_vocab": build_experimental_pytext_script_pipeline,
    "legacy_pytext_vocab": build_legacy_pytext_vocab_pipeline,
    "legacy_pytext_script_vocab": build_legacy_pytext_script_vocab_pipeline,
    "legacy_batch_torchtext": build_legacy_batch_torchtext_vocab_pipeline,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data procesing pipelines")
    parser.add_argument("--pipeline", type=str, default="sentencepiece", help="The name of pipeline")
    parser.add_argument("--dataset", type=str, default="AG_NEWS", help="Dataset for performance benchmark")
    parser.add_argument(
        "--spm-filename", type=str, default="text_unigram_25000", help="The filename of sentencepiece model"
    )
    parser.add_argument("--vocab-filename", type=str, default="vocab.txt", help="The name of vocab filename")
    args = parser.parse_args()

    if args.pipeline not in PIPELINES:
        raise KeyError(
            "Pipeline {} is not supported. Valid pipelines are {}".format(args.pipeline, list(PIPELINES.keys()))
        )

    pipeline, torchbind_pipeline, jit_pipeline = PIPELINES[args.pipeline](args)
    if pipeline is not None:
        print("Test eager mode for pipeline with pybind", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == "legacy_batch_torchtext":
            run_batch_benchmark_lookup(train, pipeline)
        else:
            run_benchmark_lookup(train, pipeline)

    if torchbind_pipeline is not None:
        print("Test eager mode for pipeline with torchbind", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == "legacy_batch_torchtext":
            run_batch_benchmark_lookup(train, torchbind_pipeline)
        else:
            run_benchmark_lookup(train, torchbind_pipeline)

    if jit_pipeline is not None:
        print("Test jit mode for pipeline", args.pipeline)
        train, test = generate_dataset(args)
        run_benchmark_lookup(train, jit_pipeline)
