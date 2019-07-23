import os
import logging
import csv
import random

import torch
import torchtext
from torchtext.datasets.text_classification import URLS

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        out = self.fc(embedded)
        return F.log_softmax(out, dim=0)


def download_data():
    data_path = '/tmp/asdf'
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp_file:
        torchtext.utils.extract_archive(torchtext.utils.download_from_url(URLS['AG_NEWS'], tmp_file.name), data_path)
    data_path = os.path.join(data_path, 'ag_news_csv')
    return os.path.join(data_path, 'train.csv'), os.path.join(data_path, 'test.csv')


def create_data(path):
    all_data = []
    with open(train_data_path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            cls = int(row[0])
            data_tokens = torchtext.datasets.text_classification.text_normalize(row[1])
            all_data.append((cls, data_tokens))
    return all_data


def build_dictionary(all_data):
    dictionary = {}
    for (cls, tokens) in all_data:
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


if __name__ == "__main__":
    num_epochs = 10
    embed_dim = 256
    batch_size = 512
    logging.basicConfig(level=logging.INFO)

    train_data_path, test_data_path = download_data()
    train_data_ = create_data(train_data_path)

    dictionary = build_dictionary(train_data_)
    print("Dictionary size: " + str(len(dictionary)))

    train_data = []
    train_labels = []
    for (cls, data) in train_data_:
        data = [dictionary.get(token, None) for token in data]
        data = list(filter(lambda x: x is not None, data))
        data = torch.tensor(data)
        if len(data) > 0:
            cls = torch.tensor([cls])
            train_data.append(data)
            train_labels.append(cls)

    model = TextSentiment(len(dictionary), embed_dim, len(set(train_labels))).cuda()
    optimizer = optim.SGD(model.parameters(), lr=1)
    loss_func = torch.nn.NLLLoss()

    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        batches = []
        for i in range(0, len(train_data), batch_size):
            data_batch = train_data[i:i+batch_size]
            text = torch.cat(data_batch)
            cls = torch.cat(train_labels[i:i+batch_size])
            offsets = [0]
            for entry in data_batch:
                offsets.append(offsets[-1] + len(entry))
            offsets = torch.tensor(offsets[:-1])
            batches.append((text, offsets, cls))
        print("Created epoch {} data".format(epoch))
        i = 0
        for (text_, offsets_, cls_) in batches:
            optimizer.zero_grad()
            text = text_.cuda()
            offsets = offsets_.cuda()
            cls = cls_.cuda()
            loss = loss_func(model(text, offsets), cls)
            loss.backward()
            optimizer.step()
            print(str(i) + "/" + str(len(batches)) + " - " + str(loss.item()))
            i += 1
