import logging
import argparse

from torchtext.datasets import text_classification
from torchtext.utils import extract_archive
from torchtext.utils import download_from_url
from torchtext.datasets.text_classification import URLS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download and extract a given dataset')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('--data', default='.data')
    parser.add_argument('--logging-level', default='WARNING')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logging_level))

    tar_file = download_from_url(URLS[args.dataset], args.data)
    extracted_files = extract_archive(tar_file, args.data)
    print("Downloaded and extracted files:")
    for extracted_file in extracted_files:
        print(extracted_file)
