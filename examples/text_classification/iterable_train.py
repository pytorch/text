import logging
import argparse

import torch
import io
import time

from torch.utils.data import DataLoader

from model import TextSentiment

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader

from tqdm import tqdm


def generate_batch(batch):
    cls = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, cls


def train(lr_, num_epoch, data_):
    data = DataLoader(
        data_,
        batch_size=batch_size,
        collate_fn=generate_batch,
        num_workers=args.num_workers,
        pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    with tqdm(unit_scale=0, unit='lines', total=train_num_lines * num_epochs) as t:
        avg_loss = 0.0
        for i, (text, offsets, cls) in enumerate(data):
            t.update(len(cls))
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
                t.set_description(
                    "lr: {:9.3f} loss: {:9.3f}".format(
                        scheduler.get_lr()[0], loss))


def test(data_):
    data = DataLoader(
        data_,
        batch_size=batch_size,
        collate_fn=generate_batch,
        num_workers=args.num_workers,
        pin_memory=True)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)
    print("Test - Accuracy: {}".format(sum(total_accuracy) / len(total_accuracy)))


def get_csv_iterator(data_path, ngrams, vocab, start=0, num_lines=None):
    def iterator(start, num_lines):
        tokenizer = get_tokenizer("basic_english")
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for i, row in enumerate(reader):
                if i == start:
                    break
            for _ in range(num_lines):
                tokens = ' '.join(row[1:])
                tokens = ngrams_iterator(tokenizer(tokens), ngrams)
                yield int(row[0]) - 1, torch.tensor([vocab[token] for token in tokens])
                try:
                    row = next(reader)
                except StopIteration:
                    f.seek(0)
                    reader = unicode_csv_reader(f)
                    row = next(reader)
    return iterator


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, iterator, num_lines, num_epochs):
        super(Dataset, self).__init__()
        self._num_lines = num_lines
        self._num_epochs = num_epochs
        self._iterator = iterator
        self._setup = False

    def _setup_iterator(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            chunk = int(self._num_lines / worker_info.num_workers)
            start = chunk * worker_info.id
            read = chunk * self._num_epochs
            if worker_info.id == worker_info.num_workers - 1:
                # The last worker needs to pick up some extra lines
                # if the number of lines aren't exactly divisible
                # by the number of workers.
                # Each epoch we loose an 'extra' number of lines.
                extra = self._num_lines % worker_info.num_workers
                extra = extra * self._num_epochs
                read += extra
        else:
            start = 0
            read = self._num_epochs * self._num_lines
        self._iterator = self._iterator(start, read)

    def __iter__(self):
        if self._setup is False:
            self._setup_iterator()
            self._setup = True
        for x in self._iterator:
            yield x


def count(data_path):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        labels = [int(row[0]) for row in reader]
        num_lines = len(labels)
        num_labels = len(set(labels))
        return num_labels, num_lines


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
    parser.add_argument('--lr-gamma', type=float, default=0.999)
    parser.add_argument('--ngrams', type=int, default=2)
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

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    logging.basicConfig(level=getattr(logging, args.logging_level))

    start_time = time.time()
    logging.info("Loading vocab from: {}".format(args.vocab))
    vocab = torch.load(args.vocab)

    logging.info("Counting training lines and labels")
    num_labels, train_num_lines = count(train_data_path)
    logging.info("Counting testing lines and labels")
    num_labels, test_num_lines = count(test_data_path)

    logging.info("Loading iterable datasets")
    train_dataset = Dataset(
        get_csv_iterator(
            train_data_path,
            ngrams,
            vocab),
        train_num_lines,
        num_epochs)
    test_dataset = Dataset(
        get_csv_iterator(
            test_data_path,
            ngrams,
            vocab),
        test_num_lines,
        num_epochs)

    logging.info("Creating models")
    model = TextSentiment(len(vocab),
                          embed_dim, num_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    logging.info("Setup took: {:3.0f}s".format(time.time() - start_time))

    logging.info("Starting training")
    train(lr, num_epochs, train_dataset)
    test(test_dataset)

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
