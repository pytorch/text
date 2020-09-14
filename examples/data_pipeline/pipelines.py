from collections import Counter, OrderedDict
import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    PyTextVocabTransform,
    PyTextScriptVocabTransform,
    iterate_batch,
    tokenizer_func,
    vocab_func,
)
from torchtext.experimental.transforms import (
    basic_english_normalize,
    TextSequentialTransforms,
    PadTransform,
)
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.functional import (
    sequential_transforms,
    totensor,
)
from torchtext.experimental.vectors import FastText as FastTextExperimental
from torchtext.experimental.vocab import vocab_from_file
from torchtext.vocab import FastText

import argparse
from torchtext.experimental.datasets.raw import text_classification as raw
import time
from dataset import BatchTextClassificationData
from torchtext.data.functional import load_sp_model


def build_sp_pipeline(spm_file):
    tokenizer = PretrainedSPTokenizer(load_sp_model(spm_file))
    vocab = PretrainedSPVocab(load_sp_model(spm_file))
    # Insert token in vocab to match a pretrained vocab
    vocab.insert_token('<pad>', 1)

    pad_id = -1
    pad_func = PadTransform(pad_id)
    pipeline = TextSequentialTransforms(tokenizer, vocab, pad_func)
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())
    print('jit sentencepiece pipeline success!')
    return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_legacy_torchtext_vocab_pipeline(vocab_file):
    tokenizer = get_tokenizer("basic_english")
    from torchtext.vocab import build_vocab_from_iterator

    def token_iterator(vocab_file):
        f = open(vocab_file, 'r')
        for line in f:
            for token in line:
                yield token

    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    pipeline = sequential_transforms(tokenizer_func(tokenizer), vocab_func(vocab))
    return iterate_batch(pipeline), None, None


def build_experimental_torchtext_pipeline(hf_vocab_file):
    tokenizer = basic_english_normalize()
    with open(hf_vocab_file, 'r') as f:
        vocab = vocab_from_file(f)
        pipeline = TextSequentialTransforms(tokenizer, vocab)
        jit_pipeline = torch.jit.script(pipeline.to_ivalue())
        print('jit experimental torchtext pipeline success!')
        return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_legacy_batch_torchtext_vocab_pipeline(vocab_file):
    tokenizer = get_tokenizer("basic_english")
    from torchtext.vocab import build_vocab_from_iterator
    from transforms import TextClassificationPipeline

    def token_iterator(vocab_file):
        f = open(vocab_file, 'r')
        for line in f:
            for token in line:
                yield token

    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    text_pipeline = sequential_transforms(tokenizer, vocab_func(vocab))
    label_pipeline = totensor(dtype=torch.long)
    return TextClassificationPipeline(label_pipeline, text_pipeline), None, None


def build_legacy_pytext_vocab_pipeline(vocab_file):
    from pytext.data.utils import Vocabulary

    tokenizer = get_tokenizer("basic_english")
    with open(vocab_file, 'r') as f:
        vocab_counter = Counter([token for line in f for token in line.rstrip()])
        sorted_by_freq_tuples = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
        vocab_list = [pair[0] for pair in sorted_by_freq_tuples]
        vocab_list.insert(0, "<unk>")
        pipeline = sequential_transforms(tokenizer_func(tokenizer),
                                         PyTextVocabTransform(Vocabulary(vocab_list, unk_token="<unk>")))
        return pipeline, None, None


def build_legacy_pytext_script_vocab_pipeline(vocab_file):
    from pytext.torchscript.vocab import ScriptVocabulary

    tokenizer = basic_english_normalize()
    with open(vocab_file, 'r') as f:
        vocab_counter = Counter([token for line in f for token in line.rstrip()])
        sorted_by_freq_tuples = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
        vocab_list = [pair[0] for pair in sorted_by_freq_tuples]
        vocab_list.insert(0, "<unk>")
        pipeline = TextSequentialTransforms(tokenizer_func(tokenizer),
                                            PyTextScriptVocabTransform(ScriptVocabulary(vocab_list)))
        jit_pipeline = torch.jit.script(pipeline.to_ivalue())
        print('jit legacy PyText pipeline success!')
        return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_experimental_pytext_script_pipeline(vocab_file):
    import os
    import sys
    # this is needed because we want to add 'torchtext/examples/vocab' directory to the
    # `sys.path` variable in order to import the pytext_vocab (since its not a module)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vocab"))
    from pytext_vocab import script_vocab

    tokenizer = basic_english_normalize()
    f = open(vocab_file, 'r')
    vocab_counter = Counter([token for line in f for token in line.rstrip()])
    ordered_dict = OrderedDict(sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True))

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer,
                                        PyTextScriptVocabTransform(script_vocab(ordered_dict)))
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())
    print('jit legacy PyText pipeline success!')
    return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_legacy_fasttext_vector_pipeline():
    tokenizer = get_tokenizer("basic_english")
    vector = FastText()

    pipeline = sequential_transforms(tokenizer_func(tokenizer), vector_func(vector))
    return pipeline, None, None


def build_experimental_fasttext_vector_pipeline():
    tokenizer = basic_english_normalize()
    vector = FastTextExperimental()

    pipeline = TextSequentialTransforms(tokenizer, vector)
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())

    print('jit legacy fasttext pipeline success!')
    return pipeline, pipeline.to_ivalue(), jit_pipeline


def run_benchmark_lookup(text_classification_dataset, pipeline):
    t0 = time.monotonic()
    lines = [text for (label, text) in text_classification_dataset]
    lines = pipeline(lines)
    print("Lookup time:", time.monotonic() - t0)


def run_batch_benchmark_lookup(text_classification_dataset, pipeline):
    t0 = time.monotonic()
    for items in text_classification_dataset:
        items = list(map(pipeline, items))
    print("Lookup time:", time.monotonic() - t0)


def generate_dataset(args):
    if args.pipeline == 'legacy_batch_torchtext':
        train = BatchTextClassificationData(args.dataset)
        test = None
    else:
        train, test = raw.DATASETS[args.dataset]()
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data procesing pipelines')
    parser.add_argument('--pipeline', type=str, default='sentencepiece',
                        help='The name of pipeline')
    parser.add_argument('--dataset', type=str, default='AG_NEWS',
                        help='Dataset for performance benchmark')
    parser.add_argument('--spm-filename', type=str, default='m_user.model',
                        help='The filename of sentencepiece model')
    parser.add_argument('--vocab-filename', type=str, default='vocab.txt',
                        help='The name of vocab filename')
    args = parser.parse_args()

    if args.pipeline == 'sentencepiece':
        pipeline, torchbind_pipeline, jit_pipeline = build_sp_pipeline(args.spm_filename)
    elif args.pipeline == 'experimental_torchtext':
        pipeline, torchbind_pipeline, jit_pipeline = build_experimental_torchtext_pipeline(args.vocab_filename)
    elif args.pipeline == 'experimental_pytext_script_vocab':
        pipeline, torchbind_pipeline, jit_pipeline = build_experimental_pytext_script_pipeline(args.vocab_filename)
    elif args.pipeline == 'experimental_fasttext':
        pipeline, torchbind_pipeline, jit_pipeline = build_experimental_fasttext_vector_pipeline()
    elif args.pipeline == 'legacy_torchtext':
        pipeline, torchbind_pipeline, jit_pipeline = build_legacy_torchtext_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'legacy_pytext_vocab':
        pipeline, torchbind_pipeline, jit_pipeline = build_legacy_pytext_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'legacy_pytext_script_vocab':
        pipeline, torchbind_pipeline, jit_pipeline = build_legacy_pytext_script_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'legacy_fasttext':
        pipeline, torchbind_pipeline, jit_pipeline = build_legacy_fasttext_vector_pipeline()
    elif args.pipeline == 'legacy_batch_torchtext':
        pipeline, torchbind_pipeline, jit_pipeline = build_legacy_batch_torchtext_vocab_pipeline(args.vocab_filename)
    else:
        print("pipeline is not supported. Current pipelines include sentencepiece, experimental_torchtext, " +
              "experimental_fasttext, legacy_pytext, experimental_fasttext, legacy_torchtext, legacy_batch_torchtext")

    if pipeline is not None:
        print("Test eager mode for pipeline with pybind", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == 'legacy_batch_torchtext':
            run_batch_benchmark_lookup(train, pipeline)
        else:
            run_benchmark_lookup(train, pipeline)

    if torchbind_pipeline is not None:
        print("Test eager mode for pipeline with torchbind", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == 'legacy_batch_torchtext':
            run_batch_benchmark_lookup(train, torchbind_pipeline)
        else:
            run_benchmark_lookup(train, torchbind_pipeline)

    if jit_pipeline is not None:
        print("Test jit mode for pipeline", args.pipeline)
        train, test = generate_dataset(args)
        run_benchmark_lookup(train, jit_pipeline)
