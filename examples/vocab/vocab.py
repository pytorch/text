import logging
import argparse

import torch
import io

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader


def csv_iterator(data_path, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            yield ngrams_iterator(tokenizer(tokens), ngrams)


parser = argparse.ArgumentParser(
    description='Train a text classification model on AG_NEWS')
parser.add_argument('--data_path', default='test.csv')
parser.add_argument('--save_vocab_path', default='saved.vocab')
parser.add_argument('--ngrams', type=int, default=2)
parser.add_argument('--logging-level', default='WARNING')
args = parser.parse_args()

ngrams = args.ngrams

logging.basicConfig(level=getattr(logging, args.logging_level))

vocab = build_vocab_from_iterator(csv_iterator(args.data_path, ngrams))

print("Saving vocab to {}".format(args.save_vocab_path))
torch.save(vocab, args.save_vocab_path)
