import os
import logging
import argparse

import torch
import sys
import io

from torchtext.datasets import text_classification
from torch.utils.data import DataLoader

from model import TextSentiment

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import build_iterable_dataset_from_iterator
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm


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
                      collate_fn=generate_batch, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    with tqdm(unit_scale=0, unit='lines', total=num_lines) as t:
        avg_loss = 0.0
        for i, (text, offsets, cls) in enumerate(data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            if i % (16 * batch_size) == 0:
                scheduler.step()
                avg_loss = avg_loss / (16 * batch_size)
                avg_loss = 0
                t.set_description("lr: {:9.3f} loss: {:9.3f}".format(scheduler.get_lr()[0], loss))
            t.update(batch_size)


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

def csv_iterator(data_path, ngrams, vocab, epochs=1, is_worker=False):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        f.seek(0, 2)
        max_offset = f.tell()
        f.seek(0)
        offset = 0.0
        if is_worker:
            offset = (max_offset / worker_info.num_workers) * worker_info.id
        f.seek(offset)
        for epoch in range(epochs):
            reader = unicode_csv_reader(f)
            for row in reader:
                tokens = ' '.join(row[1:])
                tokens = ngrams_iterator(tokenizer(tokens), ngrams)
                yield int(row[0]) - 1, torch.tensor([vocab[token] for token in tokens])
            f.seek(0)

def count_labels(data_path):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        return len(set([int(row[0]) for row in reader]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
    parser.add_argument('train_data_path')
    parser.add_argument('test_data_path')
    parser.add_argument('vocab')
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=64.0)
    parser.add_argument('--ngrams', type=int, default=2)
    parser.add_argument('--num-labels', type=int)
    parser.add_argument('--num-lines', type=int) # Optional for better progress display
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data', default='.data')
    parser.add_argument('--save-model-path')
    parser.add_argument('--logging-level', default='WARNING')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data = args.data
    ngrams = args.ngrams
    num_labels = args.num_labels
    num_lines = args.num_lines
    if num_lines:
        num_lines *= num_epochs

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    logging.basicConfig(level=getattr(logging, args.logging_level))

    logging.info("Loading vocab from: {}".format(args.vocab))
    vocab = torch.load(args.vocab)

    logging.info("Loading iterable datasets")
    train_dataset = build_iterable_dataset_from_iterator(csv_iterator(train_data_path, ngrams, vocab))
    test_dataset = build_iterable_dataset_from_iterator(csv_iterator(test_data_path, ngrams, vocab))

    if not num_labels:
        logging.info("Counting labels")
        num_labels = count_labels(test_data_path)
    logging.info("Creating models")
    model = TextSentiment(len(vocab),
                          embed_dim, num_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    logging.info("Starting training")
    train(lr, num_epochs, train_dataset)
    test(test_dataset)

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
