import os
import logging
import csv
import random
import time

import torch
import torchtext

from torchtext.datasets.text_classification import URLS
from torchtext.data.utils import generate_ngrams

def build_dictionary(all_data):
    dictionary = {}
    for tokens in all_data:
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = 1
            else:
                dictionary[token] += 1
    most_frequent_tokens = sorted(list(dictionary.items()), key=lambda x: -x[1])
    dictionary = {}
    for (token, frequency) in most_frequent_tokens:
        dictionary[token] = len(dictionary)
    return dictionary

def download_data():
    data_path = '/tmp/asdf'
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp_file:
        torchtext.utils.extract_archive(torchtext.utils.download_from_url(URLS['AG_NEWS'], tmp_file.name), data_path)
    data_path = os.path.join(data_path, 'ag_news_csv')
    return os.path.join(data_path, 'train.csv'), os.path.join(data_path, 'test.csv')

def apply_dictionary(dictionary, strings):
    for i in range(len(strings)):
        strings[i] = torch.tensor([dictionary.get(entry, dictionary['UNK']) for entry in strings[i]])

def create_data(path):
    data = []
    labels = []
    with open(train_data_path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            cls = int(row[0]) - 1
            data_tokens = torchtext.datasets.text_classification.text_normalize(row[1])
            data_tokens = generate_ngrams(data_tokens, 2)
            data.append(data_tokens)
            labels.append(cls)
    return data, labels

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_data_path, test_data_path = download_data()
    train_data, train_labels = create_data(train_data_path)
    test_data, test_labels = create_data(test_data_path)
    dictionary = build_dictionary(train_data)
    dictionary['UNK'] = len(dictionary)

    apply_dictionary(dictionary, train_data)
    apply_dictionary(dictionary, test_data)

    torch.save(train_data, "/tmp/asdf/train_data.torch")
    torch.save(train_labels, "/tmp/asdf/train_labels.torch")

    torch.save(test_data, "/tmp/asdf/test_data.torch")
    torch.save(test_labels, "/tmp/asdf/test_labels.torch")

    torch.save(dictionary, "/tmp/asdf/dictionary.torch")
