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

r"""
This example shows how to build an iterable dataset from the iterator. The
get_csv_iterator() function is used to read CSV file for the data. An abstract
dataset class setups the iterators for training the model.
"""


def generate_batch(batch):
    """
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        label: a tensor saving the labels of individual text entries.
    """

    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


r"""
torch.utils.data.DataLoader is recommended for PyTorch users to load data.
We use DataLoader here to load datasets and send it to the train()
and text() functions.
"""


def train_and_valid(lr_, num_epoch, train_data_, valid_data_):
    r"""
    Here we use SGD optimizer to train the model.

    Arguments:
        lr_: learning rate
        num_epoch: the number of epoches for training the model
        train_data_: the data used to train the model
        valid_data_: the data used to validation
        trian_len: the length of training dataset.
    """
    train_data = DataLoader(
        train_data_,
        batch_size=batch_size,
        collate_fn=generate_batch,
        num_workers=args.num_workers,
        pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_gamma)

    for epoch in range(num_epochs):

        print("Training on epoch {}".format(epoch))
        # Train the model
        with tqdm(unit_scale=0, unit='lines', total=train_len) as t:
            avg_loss = 0.0
            for i, (text, offsets, label) in enumerate(train_data):
                t.update(len(label))
                optimizer.zero_grad()
                text, offsets, label = text.to(device), offsets.to(device), \
                    label.to(device)
                output = model(text, offsets)
                loss = criterion(output, label)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                if i % (16 * batch_size) == 0:
                    avg_loss = avg_loss / (16 * batch_size)
                    avg_loss = 0
                    t.set_description(
                        "lr: {:9.3f} loss: {:9.3f}".format(
                            scheduler.get_lr()[0], loss))

        # Adjust the learning rate
        scheduler.step()

        # Test the model on valid set
        print("Valid - Accuracy: {}".format(test(valid_data_)))


def test(data_):
    r"""
    Arguments:
        data_: the data used to train the model
    """
    data = DataLoader(
        data_,
        batch_size=batch_size,
        collate_fn=generate_batch,
        num_workers=args.num_workers,
        pin_memory=True)
    total_accuracy = []
    for text, offsets, label in data:
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == label).float().mean().item()
            total_accuracy.append(accuracy)

    # In case that nothing in the dataset
    if total_accuracy == []:
        return 0.0

    return sum(total_accuracy) / len(total_accuracy)


def get_csv_iterator(data_path, ngrams, vocab, start=0, num_lines=None):
    r"""
    Generate an iterator to read CSV file.
    The yield values are an integer for the label and a tensor for the text part.

    Arguments:
        data_path: a path for the data file.
        ngrams: the number used for ngrams.
        vocab: a vocab object saving the string-to-index information
        start: the starting line to read (Default: 0). This is useful for
            on-fly multi-processing data loading.
        num_lines: the number of lines read by the iterator (Default: None).

    """
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
    r"""
    An iterable dataset to save the data. This dataset supports multi-processing
    to load the data.

    Arguments:
        iterator: the iterator to read data.
        num_lines: the number of lines read by the individual iterator.
    """
    def __init__(self, iterator, num_lines):
        super(Dataset, self).__init__()
        self._num_lines = num_lines
        self._iterator = iterator
        self._setup = False

    def _setup_iterator(self):
        r"""
        _setup_iterator() function assign the starting line and the number
        of lines to read for the individual worker. Then, send them to the iterator
        to load the data.

        If worker info is not avaialble, it will read all the lines across epochs.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            chunk = int(self._num_lines / worker_info.num_workers)
            start = chunk * worker_info.id
            read = chunk
            if worker_info.id == worker_info.num_workers - 1:
                # The last worker needs to pick up some extra lines
                # if the number of lines aren't exactly divisible
                # by the number of workers.
                # Each epoch we loose an 'extra' number of lines.
                extra = self._num_lines % worker_info.num_workers
                read += extra
        else:
            start = 0
            read = self._num_lines
        self._iterator = self._iterator(start, read)

    def __iter__(self):
        if self._setup is False:
            self._setup_iterator()
            self._setup = True
        for x in self._iterator:
            yield x


def count(data_path):
    r"""
    return the total numerber of text entries and labels.
    """
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        labels = [int(row[0]) for row in reader]
        num_lines = len(labels)
        num_labels = len(set(labels))
        return num_labels, num_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
    parser.add_argument('train_data_path', help='path for train data')
    parser.add_argument('test_data_path', help='path for test data')
    parser.add_argument('vocab', help='path for vocab object')
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='num epochs (default=5)')
    parser.add_argument('--embed-dim', type=int, default=32,
                        help='embed dim. (default=32)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--split-ratio', type=float, default=0.95,
                        help='train/valid split ratio (default=0.95)')
    parser.add_argument('--lr', type=float, default=4.0,
                        help='learning rate (default=4.0)')
    parser.add_argument('--lr-gamma', type=float, default=0.9,
                        help='gamma value for lr (default=0.9)')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='ngrams (default=2)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='num of workers (default=1)')
    parser.add_argument('--device', default='cpu',
                        help='device (default=cpu)')
    parser.add_argument('--data', default='.data',
                        help='data directory (default=.data)')
    parser.add_argument('--save-model-path',
                        help='path for saving model')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data = args.data
    ngrams = args.ngrams
    split_ratio = args.split_ratio

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

    # Split training dataset into train and valid
    train_len = int(train_num_lines * split_ratio)

    logging.info("Loading iterable datasets")
    train_dataset = Dataset(
        get_csv_iterator(
            train_data_path,
            ngrams,
            vocab, start=0, num_lines=train_len),
        train_len)

    valid_dataset = Dataset(
        get_csv_iterator(
            train_data_path,
            ngrams,
            vocab, start=train_len),
        train_num_lines - train_len)

    test_dataset = Dataset(
        get_csv_iterator(
            test_data_path,
            ngrams,
            vocab),
        test_num_lines)

    logging.info("Creating models")
    model = TextSentiment(len(vocab),
                          embed_dim, num_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    logging.info("Setup took: {:3.0f}s".format(time.time() - start_time))

    logging.info("Starting training")
    train_and_valid(lr, num_epochs, train_dataset, valid_dataset)
    print("Test - Accuracy: {}".format(test(test_dataset)))

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
