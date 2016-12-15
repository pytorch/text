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
            if x[-1] == '':
                x = x[:-1]
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


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, format, fields):

        class Example:

            def __init__(self, data):
                if format == 'json':
                    data = json.loads(data)
                if format in ('json', 'dict'):
                    for key, val in data.items():
                        if key in fields:
                            name, field = fields[key]
                            if field is not None:
                                self.__dict__[name] = field.preprocess(val)
                    return
                if data[-1] == '\n':
                    data = data[:-1]
                if format == 'tsv':
                    data = data.split('\t')
                elif format == 'csv':
                    data = list(csv.reader([data]))[0]
                for (name, field), val in zip(fields, data):
                    if field is not None:
                        self.__dict__[name] = field.preprocess(val)

        with open(os.path.expanduser(path)) as f:
            self.examples = [Example(line) for line in f]

        if isinstance(fields, dict):
            self.fields = dict(fields.values())
        else:
            self.fields = dict(fields)

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

    def __init__(self, dataset, batch_size, key, device=None, train=True,
                 repeat=True):
        self.length = math.ceil(len(dataset) / batch_size)
        self.batch_size, self.train, self.data = batch_size, train, dataset
        self.iterations, self.repeat, self.key = 0, repeat, key
        self.device = device
        if self.train:
            self.order = torch.randperm(len(self.data))

    def init(self):
        if self.train:
            xs = [self.data[i] for i in self.order]
            self.batches = pool(xs, self.batch_size, self.key)
        else:
            self.iterations = 0
            self.batches = batch(sorted(self.data, key=self.key),
                                 self.batch_size)

    @property
    def epoch(self):
        return self.iterations / self.length

    def __iter__(self):
        while True:
            self.init()
            for i, minibatch in enumerate(self.batches):
                if i == self.iterations % self.length:
                    self.iterations += 1
                    yield Batch(self.data, minibatch, self.device, self.train)
            if not self.repeat:
                raise StopIteration
            self.order = torch.randperm(len(self.data))
