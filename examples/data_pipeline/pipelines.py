import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    VocabTransform,
    PyTextVocabTransform,
    VectorTransform,
    ToLongTensor,
)
from torchtext.experimental.transforms import (
    basic_english_normalize,
    TextSequentialTransforms,
)
from torchtext.experimental.vocab import vocab_from_file_object
from torchtext.experimental.vectors import FastText
import argparse
from torchtext.experimental.datasets.raw import text_classification as raw
import time
from dataset import BatchTextClassificationData


def build_sp_pipeline(spm_file):
    tokenizer = PretrainedSPTokenizer(spm_file)
    vocab = PretrainedSPVocab(spm_file)

    # Insert token in vocab to match a pretrained vocab
    vocab.insert_token('<pad>', 1)
    pipeline = TextSequentialTransforms(tokenizer, vocab, ToLongTensor())
    jit_pipeline = torch.jit.script(pipeline)
    print('jit sentencepiece pipeline success!')
    return pipeline, jit_pipeline


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
    return pipeline, None


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
    return TextClassificationPipeline(label_pipeline, text_pipeline), None


def build_text_vocab_pipeline(hf_vocab_file):
    tokenizer = basic_english_normalize()
    f = open(hf_vocab_file, 'r')
    vocab = vocab_from_file_object(f)

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer, VocabTransform(vocab), ToLongTensor())
    torchbine_pipeline = TextSequentialTransforms(tokenizer.to_ivalue(), VocabTransform(vocab.to_ivalue()), ToLongTensor())
    jit_pipeline = torch.jit.script(torchbine_pipeline)
    print('jit Hugging Face pipeline success!')
    return pipeline, jit_pipeline


def build_pytext_vocab_pipeline(vocab_file):
    from pytext.torchscript.vocab import ScriptVocabulary
    tokenizer = BasicEnglishNormalize()
    f = open(vocab_file, 'r')
    vocab_list = [line.rstrip() for line in f]

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer,
                                        PyTextVocabTransform(ScriptVocabulary(vocab_list)),
                                        ToLongTensor())
    jit_pipeline = torch.jit.script(pipeline)
    print('jit PyText pipeline success!')
    return pipeline, jit_pipeline


def build_fasttext_vector_pipeline():
    tokenizer = BasicEnglishNormalize()
    vector = FastText()

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextSequentialTransforms(tokenizer, VectorTransform(vector))
    jit_pipeline = torch.jit.script(pipeline)
    print('jit fasttext pipeline success!')
    return pipeline, jit_pipeline


def run_benchmark_lookup(text_classification_dataset, pipeline):
    t0 = time.monotonic()
    for (label, text) in text_classification_dataset:
        text = pipeline(text)
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
        pipeline, jit_pipeline = build_sp_pipeline(args.spm_filename)
    elif args.pipeline == 'text_vocab':
        pipeline, jit_pipeline = build_text_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'pytext':
        pipeline, jit_pipeline = build_pytext_vocab_pipeline(args.vocab_filename)
    elif args.pipeline == 'fasttext':
        pipeline, jit_pipeline = build_fasttext_vector_pipeline()
    elif args.pipeline == 'torchtext':
        pipeline, jit_pipeline = build_torchtext_vocab(args.vocab_filename)
    elif args.pipeline == 'batch_torchtext':
        pipeline, jit_pipeline = build_batch_torchtext_vocab(args.vocab_filename)
    else:
        print("pipeline is not supported. Current pipelines include sentencepiece, text_vocab, " +
              "fasttext, pytext, fasttext, torchtext, batch_torchtext")

    if pipeline is not None:
        print("Test eager mode for pipeline", args.pipeline)
        train, test = generate_dataset(args)
        if args.pipeline == 'batch_torchtext':
            run_batch_benchmark_lookup(train, pipeline)
        else:
            run_benchmark_lookup(train, pipeline)

    if jit_pipeline is not None:
        print("Test jit mode for pipeline", args.pipeline)
        train, test = generate_dataset(args)
        run_benchmark_lookup(train, jit_pipeline)
