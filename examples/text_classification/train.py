import os
import logging
import argparse

import torch
import sys

from torchtext.datasets import text_classification
from torch.utils.data import DataLoader

from model import TextSentiment


def generate_batch(batch):

    def generate_offsets(data_batch):
        offsets = [0]
        for entry in data_batch:
            offsets.append(offsets[-1] + len(entry))
        offsets = torch.tensor(offsets[:-1])
        return offsets

    cls = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = generate_offsets(text)
    text = torch.cat(text)
    return text, offsets, cls


def train(lr_, num_epoch, data_):
    data = DataLoader(data_, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch, num_workers=args.num_workers)
    num_lines = num_epochs * len(data)
    for epoch in range(num_epochs):
        for i, (text, offsets, cls) in enumerate(data):
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss.backward()
            processed_lines = i + len(data) * epoch
            progress = processed_lines / float(num_lines)
            lr = lr_ * (1 - progress)
            # SGD
            for p in model.parameters():
                p.data.add_(p.grad.data * -lr)
                p.grad.detach_()
                p.grad.zero_()
            if processed_lines % 1024:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
                        progress * 100, lr, loss))
    print("")


def test(data_):
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)
    print("Test - Accuracy: {}".format(sum(total_accuracy) / len(total_accuracy)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a text classification model on AG_NEWS')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=64.0)
    parser.add_argument('--ngrams', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data', default='.data')
    parser.add_argument('--save-model-path')
    parser.add_argument('--load-vocab-path')
    parser.add_argument('--logging-level', default='WARNING')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data = args.data
    vocab = args.load_vocab_path

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(data):
        print("Creating directory {}".format(data))
        os.mkdir(data)

    if args.load_vocab_path:
        vocab = torch.load(args.load_vocab_path)

    train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
        root=data, ngrams=args.ngrams, vocab=vocab)

    model = TextSentiment(len(train_dataset.get_vocab()),
                          embed_dim, len(train_dataset.get_labels())).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train(lr, num_epochs, train_dataset)
    test(test_dataset)

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
