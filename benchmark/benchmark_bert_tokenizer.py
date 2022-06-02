from argparse import ArgumentParser

from benchmark.utils import Timer
from tokenizers import Tokenizer as hf_tokenizer_lib
from torchtext.datasets import EnWik9
from torchtext.transforms import BERTTokenizer as tt_bert_tokenizer
from transformers import BertTokenizer as hf_bert_tokenizer_slow


VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"


def benchmark_bert_tokenizer(args):
    tt_tokenizer = tt_bert_tokenizer(VOCAB_FILE, return_tokens=True)
    hf_tokenizer_slow = hf_bert_tokenizer_slow.from_pretrained("bert-base-uncased")
    hf_tokenizer_fast = hf_tokenizer_lib.from_pretrained("bert-base-uncased")
    dp = EnWik9().header(args.num_samples)
    samples = list(dp)

    with Timer("Running TorchText BERT Tokenizer on non-batched input"):
        for s in samples:
            tt_tokenizer(s)

    with Timer("Running HF BERT Tokenizer (slow) on non-batched input"):
        for s in samples:
            hf_tokenizer_slow.tokenize(s)

    with Timer("Running HF BERT Tokenizer (fast) on non-batched input"):
        for s in samples:
            hf_tokenizer_fast.encode(s)

    with Timer("Running TorchText BERT Tokenizer on batched input"):
        tt_tokenizer(samples)

    with Timer("Running HF BERT Tokenizer (fast) on batched input"):
        hf_tokenizer_fast.encode_batch(samples)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-samples", default=1000, type=int)
    benchmark_bert_tokenizer(parser.parse_args())
