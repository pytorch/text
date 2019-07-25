import os
import logging
import csv
import random
import time
import argparse

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
    parser = argparse.ArgumentParser(description='Train a text classification model on AG_NEWS')
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data', default='./data')
    parser.add_argument('--save-model-path')
    parser.add_argument('--save-dictionary-path')
    parser.add_argument('--logging-level', default='WARNING')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    device = args.device
    data = args.data

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(data):
        print("Creating directory {}".format(data))
        os.mkdir(data)

    dataset = AG_NEWS(root=data, ngrams=2)
    model = TextSentiment(len(dataset.dictionary), embed_dim, len(set(dataset.train_labels))).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=4.0)

    for epoch in range(num_epochs):
        print("Epoch: {} - Loss: {}".format(epoch,  str(train(epoch, dataset.train_data, dataset.train_labels))))
    print("Test - Accuracy: {}".format(test(dataset.test_data, dataset.test_labels)))
    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
    if args.save_dictionary_path:
        print("Saving dictionary to {}".format(args.save_dictionary_path))
        torch.save(dataset.dictionary, args.save_dictionary_path)
