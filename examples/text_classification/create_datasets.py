import os
import logging
import argparse

import torch
import sys

from torchtext.datasets import text_classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create list of Tensors for training and testing based on given datasets')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('--logging-level', default='WARNING')
    parser.add_argument('--ngrams', type=int, default=2)
    parser.add_argument('--root', default = '.data')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logging_level))
    train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
        root=args.root, ngrams=args.ngrams)
    train_data_path = os.path.join(args.root, args.dataset + "_ngrams_{}_train.data".format(args.ngrams))
    test_data_path = os.path.join(args.root, args.dataset + "_ngrams_{}_test.data".format(args.ngrams))
    print("Saving train data to {}".format(train_data_path))
    torch.save(train_dataset, train_data_path)
    print("Saving test data to {}".format(test_data_path))
    torch.save(test_dataset, test_data_path)
