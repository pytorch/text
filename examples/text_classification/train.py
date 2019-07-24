import os
import logging
import csv
import random
import time

import torch
import torchtext

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchtext.datasets.text_classification import AG_NEWS

from model import TextSentiment

def generate_offsets(data_batch):
    offsets = [0]
    for entry in data_batch:
        offsets.append(offsets[-1] + len(entry))
    offsets = torch.tensor(offsets[:-1])
    return offsets

def generate_batch(data, labels, i, batch_size):
    data_batch = data[i:i+batch_size]
    text = torch.cat(data_batch)
    offsets = generate_offsets(data_batch)
    cls = torch.tensor(labels[i:i+batch_size])
    text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
    return text, offsets, cls

def train(epoch, data, labels):
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    labels = [labels[i] for i in perm]

    total_loss = []
    for i in range(0, len(data), batch_size):
        text, offsets, cls = generate_batch(data, labels, i, batch_size)
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, cls)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(total_loss) / len(total_loss)

def test(data, labels):
    total_accuracy = []
    for i in range(0, len(data), batch_size):
        text, offsets, cls = generate_batch(data, labels, i, batch_size)
        output = model(text, offsets)
        accuracy = (output.argmax(1) == cls).float().mean()
        total_accuracy.append(accuracy)
    return torch.tensor(total_accuracy).float().mean()

if __name__ == "__main__":
    num_epochs = 3
    embed_dim = 128
    batch_size = 512
    device = 'cuda:1'

    dataset = AG_NEWS(ngrams=2)
    model = TextSentiment(len(dataset.dictionary), embed_dim, len(set(dataset.train_labels))).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=4.0)

    for epoch in range(num_epochs):
        print("Epoch: {} - Loss: {}".format(epoch,  str(train(epoch, dataset.train_data, dataset.train_labels))))
    print("Test accuracy: {}".format(test(dataset.test_data, dataset.test_labels)))
    torch.save(model.to('cpu'), "/tmp/asdf/model.torch")
