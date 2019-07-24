import os
import logging
import csv
import random
import time

import torch
import torchtext
import tempfile

from torchtext.datasets.text_classification import URLS
from torchtext.data.utils import generate_ngrams
from collections import Counter
from collections import OrderedDict
from torchtext.utils import extract_archive
from torchtext.utils import download_from_url

def build_dictionary_from_path(data_path):
    dictionary = Counter()
    with open(data_path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            tokens = torchtext.datasets.text_classification.text_normalize(row[1])
            tokens = generate_ngrams(tokens, 2)
            dictionary.update(tokens)
    word_dictionary = OrderedDict()
    for (token, frequency) in dictionary.most_common():
        word_dictionary[token] = len(word_dictionary)
    return word_dictionary

def create_data(dictionary, path):
    data = []
    labels = []
    with open(train_data_path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            cls = int(row[0]) - 1
            tokens = torchtext.datasets.text_classification.text_normalize(row[1])
            tokens = generate_ngrams(tokens, 2)
            tokens = torch.tensor([dictionary.get(entry, dictionary['UNK']) for entry in tokens])
            data.append(tokens)
            labels.append(cls)
    return data, labels

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_path = '/tmp/asdf'
    data_path = os.path.join(data_path, 'amazon_review_full_csv')
    train_data_path = os.path.join(data_path, 'train.csv')
    test_data_path = os.path.join(data_path, 'test.csv')

    print("Building dictionary")
    dictionary = build_dictionary_from_path(train_data_path)
    dictionary['UNK'] = len(dictionary)
    print("Dictionary size: " + str(len(dictionary)))

    print("Creating data")
    train_data, train_labels = create_data(dictionary, train_data_path)
    test_data, test_labels = create_data(dictionary, test_data_path)

    print("Saving data")
    torch.save(train_data, "/tmp/asdf/train_data.torch")
    torch.save(train_labels, "/tmp/asdf/train_labels.torch")

    torch.save(test_data, "/tmp/asdf/test_data.torch")
    torch.save(test_labels, "/tmp/asdf/test_labels.torch")

    print("Saving dictionary")
    torch.save(dictionary, "/tmp/asdf/dictionary.torch")
