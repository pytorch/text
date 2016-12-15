import torch
import torch.utils.data
from torch.autograd import Variable
from text.torchtext.vocab import Vocab

from collections import Counter
import csv
import json
import math
import os
import random

class Pipeline:

    def __call__(self, x, *args):
        if isinstance(x, list):
            return [self.__call__(tok, *args) for tok in x]
        return self.convert_token(x, *args)

    def convert_token(self, token, *args):
        return token


class Field:

    def __init__(
            self, time_series=False, use_vocab=True, init_token=None,
            eos_token=None, fix_length=None, tensor_type=torch.LongTensor,
            before_numericalizing=Pipeline(), after_numericalizing=Pipeline(),
            tokenize=(lambda x: x.split(' '))):
        self.time_series = time_series
        self.use_vocab = use_vocab
        self.fix_length = fix_length
        self.init_token = init_token
        self.eos_token = eos_token
        self.tokenize = tokenize
        self.before_numericalizing = before_numericalizing
        self.after_numericalizing = after_numericalizing
        self.tensor_type = tensor_type

    def preprocess(self, x):
        if self.time_series and isinstance(x, str):
            x = self.tokenize(x)
        return x

    def pad(self, batch):
        batch = list(batch)
        if not self.time_series:
            return batch
        if self.fix_length is None:
            max_len = max(len(x) for x in batch)
        else:
            max_len = self.fix_length
        max_len = max_len + (self.init_token, self.eos_token).count(None) - 2
        padded = []
        for x in batch:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                ['<pad>'] * max(0, max_len - len(x)))
        return padded

    def build_vocab(self, *args, lower=False, **kwargs):
        counter = Counter()
        for data in args:
            for x in data:
                if not self.time_series:
                    x = [x]
                if lower:
                    x = [token.lower() for token in x]
                counter.update(x)
        specials = ['<pad>', self.init_token, self.eos_token]
        specials = [token for token in specials if token is not None]
        self.vocab = Vocab(counter, specials=specials, lower=lower, **kwargs)

    def numericalize(self, arr, device=None, train=True):
        if self.use_vocab:
            arr = self.before_numericalizing(arr, self.vocab, train)
            if self.time_series:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]
            arr = self.after_numericalizing(arr, self.vocab, train)
        else:
            arr = self.after_numericalizing(arr, train)
        if device == -1:
            arr = self.tensor_type(arr)
        else:
            with torch.cuda.device(device):
                arr = self.tensor_type(arr).cuda()
        return Variable(arr, volatile=not train)


class Example:

    @classmethod
    def fromJSON(cls, data, fields):
        return cls.fromdict(json.loads(data), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, val in data.items():
            if key in fields:
                name, field = fields[key]
                if field is not None:
                    setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromTSV(cls, data, fields):
        if data[-1] == '\n':
            data = data[:-1]
        return cls.fromlist(data.split('\t'), fields)

    @classmethod
    def fromCSV(cls, data, fields):
        if data[-1] == '\n':
            data = data[:-1]
        return cls.fromlist(list(csv.reader([data]))[0])

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
        return ex


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, format, fields):

        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromTSV, 'csv': Example.fromCSV}[format.lower()]

        with open(os.path.expanduser(path)) as f:
            self.examples = [make_example(line, fields) for line in f]

        if isinstance(fields, dict):
            self.fields = dict(fields.values())
        else:
            self.fields = dict(fields)

    @classmethod
    def splits(cls, path, train=None, dev=None, test=None, **kwargs):
        train_data = None if train is None else cls(path + train, **kwargs)
        dev_data = None if dev is None else cls(path + dev, **kwargs)
        test_data = None if test is None else cls(path + test, **kwargs)
        return tuple(d for d in (train_data, dev_data, test_data)
                     if d is not None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        yield from self.examples

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


def batch(data, batch_size):
    minibatch = []
    for ex in data:
        minibatch.append(ex)
        if len(minibatch) == batch_size:
            yield minibatch
            minibatch = []
    if minibatch:
        yield minibatch

def shuffle(data):
    data = list(data)
    random.shuffle(data)
    return data

def pool(data, batch_size, key):
    for p in batch(data, batch_size * 100):
        yield from shuffle(batch(sorted(p, key=key), batch_size))


class Batch:

    def __init__(self, dataset, data, device=None, train=True):
        self.batch_size = len(data)
        self.dataset = dataset
        self.train = train
        for (name, field) in dataset.fields.items():
            if field is not None:
                self.__dict__[name] = field.numericalize(
                    field.pad(x.__dict__[name] for x in data),
                    device=device, train=train)


class BucketIterator:

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 train=True, repeat=None):
        self.length = math.ceil(len(dataset) / batch_size)
        self.batch_size, self.train, self.data = batch_size, train, dataset
        self.iterations, self.repeat = 0, train if repeat is None else repeat
        if sort_key is None:
            try:
                self.sort_key = dataset.sort_key
            except AttributeError:
                print('Must provide sort_key with constructor or dataset')
        else:
            self.sort_key = sort_key
        self.device = device
        if self.train:
            self.order = torch.randperm(len(self.data))

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            train = i == 0
            ret.append(cls(
                datasets[i], batch_size=batch_sizes[i], train=train, **kwargs))
        return tuple(ret)

    def init_epoch(self):
        if self.train:
            xs = [self.data[i] for i in self.order]
            self.batches = pool(xs, self.batch_size, self.sort_key)
        else:
            self.iterations = 0
            self.batches = batch(sorted(self.data, key=self.sort_key),
                                 self.batch_size)

    @property
    def epoch(self):
        return self.iterations / self.length

    def __iter__(self):
        while True:
            self.init_epoch()
            for i, minibatch in enumerate(self.batches):
                if i == self.iterations % self.length:
                    self.iterations += 1
                    yield Batch(self.data, minibatch, self.device, self.train)
            if not self.repeat:
                raise StopIteration
            self.order = torch.randperm(len(self.data))
