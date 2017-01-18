import torch
import torch.utils.data
from torch.autograd import Variable
from .vocab import Vocab

from collections import Counter
from collections import OrderedDict
import csv
import json
import math
import os
import random
from six.moves import urllib
import zipfile

class Pipeline:

    def __call__(self, x, *args):
        if isinstance(x, list):
            return [self.__call__(tok, *args) for tok in x]
        return self.convert_token(x, *args)

    def convert_token(self, token, *args):
        return token


def get_tokenizer(tokenizer):
    if not isinstance(tokenizer, str):
        return tokenizer
    if tokenizer == 'spacy':
        try:
            import spacy
            spacy_en = spacy.load('en')
            return lambda s: [tok.text for tok in spacy_en.tokenize(s)]
        except ImportError:
            print('''Please install SpaCy and the SpaCy English tokenizer:
    $ conda install libgcc
    $ pip install spacy
    $ python -m spacy.en.download tokenizer''')
            raise
        except AttributeError:
            print('''Please install the SpaCy English tokenizer:
    $ python -m spacy.en.download tokenizer''')
            raise


class Field:

    def __init__(
            self, time_series=False, use_vocab=True, init_token=None,
            eos_token=None, fix_length=None, tensor_type=torch.LongTensor,
            before_numericalizing=Pipeline(), after_numericalizing=Pipeline(),
            tokenize=(lambda s: s.split())):
        self.time_series = time_series
        self.use_vocab = use_vocab
        self.fix_length = fix_length
        self.init_token = init_token
        self.eos_token = eos_token
        self.tokenize = get_tokenizer(tokenize)
        self.before_numericalizing = before_numericalizing
        self.after_numericalizing = after_numericalizing
        self.tensor_type = tensor_type

    def preprocess(self, x):
        if self.time_series and isinstance(x, str):
            x = self.tokenize(x)
        return x

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.time_series:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded = []
        for x in minibatch:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                ['<pad>'] * max(0, max_len - len(x)))
        return padded

    def build_vocab(self, *args, lower=False, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.time_series:
                    x = [x]
                if lower:
                    x = [token.lower() for token in x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(tok for tok in [
            '<pad>', self.init_token, self.eos_token] if tok is not None))
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
        arr = self.tensor_type(arr)
        if self.time_series:
            arr.t_()
        if device == -1:
            if self.time_series:
                arr = arr.contiguous()
        else:
            with torch.cuda.device(device):
                arr = arr.cuda()
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

    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):

        if filter_pred is not None:
            examples = list(filter(filter_pred, examples))
        self.examples = examples

        if isinstance(fields, dict):
            fields = fields.values()
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
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        yield from self.examples

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ZipDataset(Dataset):

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            zpath = os.path.join(root, cls.filename)
            if not os.path.isfile(zpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, zpath)
            with zipfile.ZipFile(zpath, 'r') as zfile:
                print('extracting')
                zfile.extractall(root)
        return path



class TabularDataset(Dataset):

    def __init__(self, path, format, fields, **kwargs):

        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromTSV, 'csv': Example.fromCSV}[format.lower()]

        with open(os.path.expanduser(path)) as f:
            examples = [make_example(line, fields) for line in f]

        super().__init__(examples, fields, **kwargs)


def batch(data, batch_size):
    minibatch = []
    for ex in data:
        minibatch.append(ex)
        if len(minibatch) == batch_size:
            yield minibatch
            minibatch = []
    if minibatch:
        yield minibatch

def shuffled(data):
    data = list(data)
    random.shuffle(data)
    return data

def pool(data, batch_size, key):
    for p in batch(data, batch_size * 100):
        yield from shuffled(batch(sorted(p, key=key), batch_size))


class Batch:

    def __init__(self, data=None, dataset=None, device=None, train=True):
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None:
                    setattr(self, name, field.numericalize(
                        field.pad(x.__dict__[name] for x in data),
                        device=device, train=train))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch


class Iterator:

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 train=True, repeat=None, shuffle=None, sort=None):
        self.batch_size, self.train, self.dataset = batch_size, train, dataset
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
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            train = i == 0
            ret.append(cls(
                datasets[i], batch_size=batch_sizes[i], train=train, **kwargs))
        return tuple(ret)

    def data(self):
        if self.shuffle:
            xs = [self.dataset[i] for i in torch.randperm(len(self.dataset))]
        elif self.sort:
            xs = sorted(self.dataset, key=self.sort_key)
        else:
            xs = self.dataset
        return xs

    def init_epoch(self):
        self.batches = batch(self.data(), self.batch_size)

    @property
    def epoch(self):
        return self.iterations / len(self)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        while True:
            self.init_epoch()
            for i, minibatch in enumerate(self.batches):
                if i == self.iterations % len(self):
                    self.iterations += 1
                    yield Batch(minibatch, self.dataset, self.device,
                                self.train)
            if not self.repeat:
                raise StopIteration

class BucketIterator(Iterator):

    def init_epoch(self):
        if self.repeat:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key)
        else:
            self.iterations = 0
            self.batches = batch(self.data(), self.batch_size)


class BPTTIterator(Iterator):

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super().__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset[0].text) /
                         (self.batch_size * self.bptt_len))

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ['<pad>'] * (math.ceil(len(text) / self.batch_size) *
                                   self.batch_size - len(text))
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


def interleave_keys(a, b):
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)
