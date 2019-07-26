import os
import logging
import argparse

import torch
import io

from torchtext.datasets import text_classification
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader
from torchtext.utils import extract_archive
from torchtext.utils import download_from_url
from torchtext.datasets.text_classification import URLS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download and extract a given dataset')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('root')
    parser.add_argument('--logging-level', default='WARNING')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logging_level))

    tar_file = download_from_url(URLS[args.dataset], args.root)
    extracted_files = extract_archive(tar_file, args.root)
    print("Downloaded and extracted files:")
    for extracted_file in extracted_files:
        print(extracted_file)

