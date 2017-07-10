import math
import random

import torch

from .batch import Batch
from .dataset import Dataset


class Iterator(object):
    """Defines an iterator that loads batches of data from a Dataset.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        batch_size_fn: Function of three arguments (new example to add, example
            index, and current effective batch size) that returns the new
            effective batch size resulting from adding that example to a batch.
            This is useful for dynamic batching, where this function would add
            to the current effective batch size the number of tokens in the new
            example.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device: Device to create batches on. Use -1 for CPU and None for the
            currently active GPU device.
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=lambda new, i, sofar: i + 1, train=True,
                 repeat=None, shuffle=None, sort=None):
        self.batch_size, self.train, self.dataset = batch_size, train, dataset
        self.batch_size_fn = batch_size_fn
        self.iterations = 0
        self.repeat = train if repeat is None else repeat
        self.shuffle = train if shuffle is None else shuffle
        self.sort = not train if sort is None else sort
        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key
        self.device = device

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        """Create Iterator objects for multiple splits of a dataset.

        Arguments:
            datasets: Tuple of Dataset objects corresponding to the splits. The
                first such object should be the train set.
            batch_sizes: Tuple of batch sizes to use for the different splits,
                or None to use the same batch_size for all splits.
            Remaining keyword arguments: Passed to the constructor of the
                iterator class being used.
        """
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            train = i == 0
            ret.append(cls(
                datasets[i], batch_size=batch_sizes[i], train=train, **kwargs))
        return tuple(ret)

    def data(self):
        """Return the examples in the dataset in order, sorted, or shuffled."""
        if self.shuffle:
            xs = [self.dataset[i] for i in torch.randperm(len(self.dataset))]
        elif self.sort:
            xs = sorted(self.dataset, key=self.sort_key)
        else:
            xs = self.dataset
        return xs

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""
        self.batches = batch(self.data(), self.batch_size, self.batch_size_fn)
        if not self.repeat:
            self.iterations = 0

    @property
    def epoch(self):
        return self.iterations / len(self)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        while True:
            self.init_epoch()
            for minibatch in self.batches:
                self.iterations += 1
                yield Batch(minibatch, self.dataset, self.device,
                            self.train)
            if not self.repeat:
                raise StopIteration


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device: Device to create batches on. Use -1 for CPU and None for the
            currently active GPU device.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset[0].text) /
                         (self.batch_size * self.bptt_len))

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * (math.ceil(len(text) / self.batch_size) *
                                           self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device, train=self.train)
        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                seq_len = min(self.bptt_len, len(data) - 1 - i)
                yield Batch.fromvars(
                    dataset, self.batch_size, train=self.train,
                    text=data[i:i + seq_len],
                    target=data[i + 1:i + 1 + seq_len])
            if not self.repeat:
                raise StopIteration


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def init_epoch(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn)
        if not self.repeat:
            self.iterations = 0


def batch(data, batch_size, batch_size_fn=lambda new, i, sofar: i + 1):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, size_so_far)
        if size_so_far >= batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
    if minibatch:
        yield minibatch


def shuffled(data):
    data = list(data)
    random.shuffle(data)
    return data


def pool(data, batch_size, key, batch_size_fn=lambda new, i, sofar: i + 1):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    for p in batch(data, batch_size * 100, batch_size_fn):
        for b in shuffled(batch(sorted(p, key=key), batch_size, batch_size_fn)):
            yield b
