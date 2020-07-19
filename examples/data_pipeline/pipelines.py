import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    TextDataPipeline,
    VocabTransform,
    VectorTransform,
)
from torchtext.experimental.transforms import (
    BasicEnglishNormalize,
)
from torchtext.experimental.vocab import vocab_from_file_object
from torchtext.experimental.vectors import FastText
import argparse


def build_sp_pipeline(spm_file):
    tokenizer = PretrainedSPTokenizer(spm_file)
    vocab = PretrainedSPVocab(spm_file)

    # Insert token in vocab to match a pretrained vocab
    vocab.insert_token('<pad>', 1)
    pipeline = TextDataPipeline(tokenizer, vocab)
    jit_pipeline = torch.jit.script(pipeline)
    print('jit sentencepiece pipeline success!')
    return pipeline, jit_pipeline


def build_huggingface_vocab_pipeline(hf_vocab_file):
    tokenizer = BasicEnglishNormalize()
    f = open(hf_vocab_file, 'r')
    vocab = vocab_from_file_object(f)

    # Insert token in vocab to match a pretrained vocab
    # pipeline = TextDataPipeline(tokenizer, vocab.lookup_indices)
    pipeline = TextDataPipeline(tokenizer, VocabTransform(vocab))
    jit_pipeline = torch.jit.script(pipeline)
    print('jit Hugging Face pipeline success!')
    return pipeline, jit_pipeline


def build_fasttext_vector_pipeline():
    tokenizer = BasicEnglishNormalize()
    vector = FastText()

    # Insert token in vocab to match a pretrained vocab
    pipeline = TextDataPipeline(tokenizer, VectorTransform(vector))
    jit_pipeline = torch.jit.script(pipeline)
    print('jit fasttext pipeline success!')
    return pipeline, jit_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data procesing pipelines')
    parser.add_argument('--pipeline', type=str, default='sentencepiece',
                        help='The name of pipeline')
    parser.add_argument('--dataset', type=str, default='AG_NEWS',
                        help='Dataset for performance benchmark')
    parser.add_argument('--spm-filename', type=str, default='m_user.model',
                        help='The filename of sentencepiece model')
    parser.add_argument('--hf-vocab-filename', type=str, default='vocab.txt',
                        help='The name of Hugging Face vocab filename')
    args = parser.parse_args()

    if args.pipeline == 'sentencepiece':
        pipeline, jit_pipeline = build_sp_pipeline(args.spm_filename)
    elif args.pipeline == 'huggingface':
        pipeline, jit_pipeline = build_huggingface_vocab_pipeline(args.hf_vocab_filename)
    elif args.pipeline == 'fasttext':
        pipeline, jit_pipeline = build_fasttext_vector_pipeline()
    else:
        print("pipeline is not supported. Current pipelines include sentencepiece, huggingface, fasttext")

    print("eager mode:", pipeline("torchtext provides building blocks for data processing"))
    print("jit mode:", jit_pipeline("torchtext provides building blocks for data processing"))
