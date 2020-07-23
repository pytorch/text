import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    TextDataPipeline,
    VocabTransform,
    PyTextVocabTransform,
    VectorTransform,
)
from torchtext.experimental.transforms import (
    BasicEnglishNormalize,
)
from torchtext.experimental.vocab import vocab_from_file_object
from torchtext.experimental.vectors import FastText
import argparse
from torchtext.experimental.datasets.raw import text_classification as raw
import time
from functools import partial
from pytext_vocab import ScriptVocabulary


def build_sp_pipeline(spm_file):
    tokenizer = PretrainedSPTokenizer(spm_file)
    vocab = PretrainedSPVocab(spm_file)

    # Insert token in vocab to match a pretrained vocab
    vocab.insert_token('<pad>', 1)
    pipeline = TextDataPipeline(tokenizer, vocab)
    jit_pipeline = torch.jit.script(pipeline)
    print('jit sentencepiece pipeline success!')
    return pipeline, jit_pipeline


def build_torchtext_vocab(vocab_file):
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")
    from torchtext.vocab import build_vocab_from_iterator

    def token_iterator(vocab_file):
        f = open(vocab_file, 'r')
        for token in f:
            yield token
    vocab = build_vocab_from_iterator(token_iterator(vocab_file))
    pipeline = TextDataPipeline(tokenizer, partial(map, vocab))
    return pipeline, None


def build_huggingface_vocab_pipeline(hf_vocab_file):
    tokenizer = BasicEnglishNormalize()
    f = open(hf_vocab_file, 'r')
    vocab = vocab_from_file_object(f)

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextDataPipeline(tokenizer, VocabTransform(vocab))
    jit_pipeline = torch.jit.script(pipeline)
    print('jit Hugging Face pipeline success!')
    return pipeline, jit_pipeline


def build_pytext_vocab_pipeline(vocab_file):
    tokenizer = BasicEnglishNormalize()
    f = open(vocab_file, 'r')
    vocab_list = [line.rstrip() for line in f]

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextDataPipeline(tokenizer,
                                PyTextVocabTransform(ScriptVocabulary(vocab_list)))
    jit_pipeline = torch.jit.script(pipeline)
    print('jit PyText pipeline success!')
    return pipeline, jit_pipeline


def build_fasttext_vector_pipeline():
    tokenizer = BasicEnglishNormalize()
    vector = FastText()

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextDataPipeline(tokenizer, VectorTransform(vector))
    jit_pipeline = torch.jit.script(pipeline)
    print('jit fasttext pipeline success!')
    return pipeline, jit_pipeline


def run_benchmark_lookup(text_classification_dataset, pipeline):
    t0 = time.monotonic()
    for (label, text) in text_classification_dataset:
        text = pipeline(text)
    print("Lookup time:", time.monotonic() - t0)


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
        pipeline, jit_pipeline = build_sp_pipeline(args.spm_filename)
    elif args.pipeline == 'huggingface':
        pipeline, jit_pipeline = build_huggingface_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'pytext':
        pipeline, jit_pipeline = build_pytext_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'fasttext':
        pipeline, jit_pipeline = build_fasttext_vector_pipeline()
    elif args.pipeline == 'torchtext':
        pipeline, jit_pipeline = build_torchtext_vocab(args.vocab_filename)
    else:
        print("pipeline is not supported. Current pipelines include sentencepiece, huggingface, fasttext")
    print("Test eager mode for pipeline", args.pipeline)
    train, test = raw.DATASETS[args.dataset]()
    if pipeline is not None:
        run_benchmark_lookup(train, pipeline)
    print("Test jit mode for pipeline", args.pipeline)
    train, test = raw.DATASETS[args.dataset]()
    if jit_pipeline is not None:
        run_benchmark_lookup(train, jit_pipeline)
