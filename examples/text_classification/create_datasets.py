import os
import logging
import argparse

import torch

from torchtext.datasets import text_classification

r"""
Once you have the datasets, you can save them as a list of tensors
for re-use. Here is an example to load/save text_classification datasets.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        'Create list of Tensors for training and '
        'testing based on given datasets'))
    parser.add_argument('dataset', choices=text_classification.DATASETS,
                        help='dataset name')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='ngrams (default=2)')
    parser.add_argument('--root', default='.data',
                        help='data directory (default=.data)')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logging_level))
    train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
        root=args.root, ngrams=args.ngrams)
    train_data_path = os.path.join(
        args.root,
        args.dataset +
        "_ngrams_{}_train.data".format(
            args.ngrams))
    test_data_path = os.path.join(
        args.root,
        args.dataset +
        "_ngrams_{}_test.data".format(
            args.ngrams))
    print("Saving train data to {}".format(train_data_path))
    torch.save(train_dataset, train_data_path)
    print("Saving test data to {}".format(test_data_path))
    torch.save(test_dataset, test_data_path)
