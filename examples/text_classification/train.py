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
    data = DataLoader(data_, batch_size=batch_size,
                      collate_fn=generate_batch, num_workers=args.num_workers)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    import time
    t0 = time.time()
    for epoch in range(num_epochs):
        for i, (text, offsets, cls) in enumerate(data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss.backward()
            optimizer.step()
            if i % 16 == 0:
                sys.stderr.write(
                        "\rEpoch: {:9.3f} lr: {:9.3f} loss: {:9.3f} Time: {:9.3f}s".format(epoch, scheduler.get_lr()[0], loss, time.time() - t0))
            if i % (16 * batch_size) == 0:
                scheduler.step()
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
        description='Train a text classification model on text classification datasets.')
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

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(data):
        print("Creating directory {}".format(data))
        os.mkdir(data)

    vocab = args.load_vocab_path

    train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
        root=data, ngrams=args.ngrams, iterable=True, vocab=vocab)

    model = TextSentiment(len(train_dataset.get_vocab()),
                          embed_dim, 4).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train(lr, num_epochs, train_dataset)
    test(test_dataset)

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
