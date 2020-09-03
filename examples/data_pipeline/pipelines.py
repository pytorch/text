import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    PyTextVocabTransform,
    iterate_batch,
)
from torchtext.experimental.transforms import (
    basic_english_normalize,
    TextSequentialTransforms,
)
from torchtext.experimental.vocab import vocab_from_file
from torchtext.experimental.vectors import FastText
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
    pipeline = TextSequentialTransforms(tokenizer, vocab)
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())
    print('jit sentencepiece pipeline success!')
    return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_torchtext_vocab(vocab_file):
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")
    from torchtext.vocab import build_vocab_from_iterator
    from torchtext.experimental.functional import totensor, vocab_func, sequential_transforms

    def token_iterator(vocab_file):
        f = open(vocab_file, 'r')
        for token in f:
            yield token
    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    pipeline = sequential_transforms(tokenizer, vocab_func(vocab), totensor(dtype=torch.long))
    return iterate_batch(pipeline), None, None


def build_batch_torchtext_vocab(vocab_file):
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")
    from torchtext.vocab import build_vocab_from_iterator
    from transforms import TextClassificationPipeline
    from torchtext.experimental.functional import totensor, vocab_func, sequential_transforms

    def token_iterator(vocab_file):
        f = open(vocab_file, 'r')
        for token in f:
            yield token
    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    text_pipeline = sequential_transforms(tokenizer, vocab_func(vocab), totensor(dtype=torch.long))
    label_pipeline = totensor(dtype=torch.long)
    return TextClassificationPipeline(label_pipeline, text_pipeline), None, None


def build_text_vocab_pipeline(hf_vocab_file):
    tokenizer = basic_english_normalize()
    f = open(hf_vocab_file, 'r')
    vocab = vocab_from_file(f)

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer, vocab)
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())
    print('jit text vocab pipeline success!')
    return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_pytext_vocab_pipeline(vocab_file):
    from pytext.torchscript.vocab import ScriptVocabulary
    tokenizer = basic_english_normalize()
    f = open(vocab_file, 'r')
    vocab_list = [line.rstrip() for line in f]

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer,
                                        PyTextVocabTransform(ScriptVocabulary(vocab_list)))
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())
    print('jit PyText pipeline success!')
    return pipeline, pipeline.to_ivalue(), jit_pipeline


def build_fasttext_vector_pipeline():
    tokenizer = basic_english_normalize()
    vector = FastText()

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer, vector)
    jit_pipeline = torch.jit.script(pipeline.to_ivalue())
    print('jit fasttext pipeline success!')
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
    if args.pipeline == 'batch_torchtext':
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
    elif args.pipeline == 'text_vocab':
        pipeline, torchbind_pipeline, jit_pipeline = build_text_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'pytext':
        pipeline, torchbind_pipeline, jit_pipeline = build_pytext_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'fasttext':
        pipeline, torchbind_pipeline, jit_pipeline = build_fasttext_vector_pipeline()
    elif args.pipeline == 'torchtext':
        pipeline, torchbind_pipeline, jit_pipeline = build_torchtext_vocab(args.vocab_filename)
    elif args.pipeline == 'batch_torchtext':
        pipeline, torchbind_pipeline, jit_pipeline = build_batch_torchtext_vocab(args.vocab_filename)
    else:
        print("pipeline is not supported. Current pipelines include sentencepiece, text_vocab, " +
              "fasttext, pytext, fasttext, torchtext, batch_torchtext")

    if pipeline is not None:
        print("Test eager mode for pipeline with pybind", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == 'batch_torchtext':
            run_batch_benchmark_lookup(train, pipeline)
        else:
            run_benchmark_lookup(train, pipeline)

    if torchbind_pipeline is not None:
        print("Test eager mode for pipeline with torchbind", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == 'batch_torchtext':
            run_batch_benchmark_lookup(train, torchbind_pipeline)
        else:
            run_benchmark_lookup(train, torchbind_pipeline)

    if jit_pipeline is not None:
        print("Test jit mode for pipeline", args.pipeline)
        train, test = generate_dataset(args)
        run_benchmark_lookup(train, jit_pipeline)
