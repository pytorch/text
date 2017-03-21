from __future__ import print_function
import six
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


def just_split(s):
    return s.split()

def identity(x):
    return x

def return_second(x):
    return x[1]

class Pipeline(object):
    """Defines a pipeline for transforming sequence data."""

    def __init__(self, convert_token=None):
        if convert_token is not None:
            self.convert_token = convert_token
        else:
            self.convert_token = identity
        self.pipes = [self]

    def __call__(self, x, *args):
        for pipe in self.pipes:
            x = pipe.call(x)
        return x

    def call(self, x, *args):
        if isinstance(x, list):
            return [self(tok, *args) for tok in x]
        return self.convert_token(x, *args)

    def add_before(self, pipeline):
        """Add `pipeline` before this processing pipeline."""
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = pipeline.pipes[:] + self.pipes[:]

    def add_after(self, pipeline):
        """Add `pipeline` after this processing pipeline."""
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = self.pipes[:] + pipeline.pipes[:]


def get_tokenizer(tokenizer):
    if not isinstance(tokenizer, str):
        return tokenizer
    if tokenizer == 'spacy':
        try:
            import spacy
            spacy_en = spacy.load('en')
            return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
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


class Field(object):
    """Defines a datatype together with instructions for converting to Tensor.

    Every dataset consists of one or more types of data. For instance, a text
    classification dataset contains sentences and their classes, while a
    machine translation dataset contains paired examples of text in two
    languages. Each of these types of data is represented by a Field object,
    which holds a Vocab object that defines the set of possible values for
    elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. Default:
            True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: the identity pipeline.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. Default: the identity pipeline.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. Default: str.split.
    """

    def __init__(
            self, sequential=True, use_vocab=True, init_token=None,
            eos_token=None, fix_length=None, tensor_type=torch.LongTensor,
            preprocessing=None, postprocessing=None, lower=False,
            tokenize=just_split, include_lengths=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.fix_length = fix_length
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = '<pad>' if self.sequential else None
        self.tokenize = get_tokenizer(tokenize)
        self.lower = lower
        self.include_lengths = include_lengths
        self.preprocessing = (Pipeline() if preprocessing
                              is None else preprocessing)
        self.postprocessing = (Pipeline() if postprocessing
                               is None else postprocessing)
        self.tensor_type = tensor_type

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary."""
        if self.sequential and isinstance(x, str):
            x = self.tokenize(x)
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        return self.preprocessing(x)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                ['<pad>'] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
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
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.pad_token, self.init_token, self.eos_token]
            if tok is not None))
        self.vocab = Vocab(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr: List of tokenized and padded examples, or tuple of a padded
                list and a list of lengths if self.include_lengths is True.
            device: Device to create the Variable's Tensor on. Use -1 for
                CPU and None for the currently active GPU device. Default:
                None.
            train: Whether the batch is for a training set. If False, the
                Variable will be created with volatile=True. Default: True.
        """
        if self.sequential and isinstance(arr[0], tuple):
            arr = tuple(zip(*arr))
        if isinstance(arr, tuple):
            arr, lengths = arr
        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]
            arr = self.postprocessing(arr, self.vocab, train)
        else:
            arr = self.postprocessing(arr, train)
        arr = self.tensor_type(arr)
        if self.sequential:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            with torch.cuda.device(device):
                arr = arr.cuda()
        if self.include_lengths:
            return Variable(arr, volatile=not train), lengths
        return Variable(arr, volatile=not train)


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def fromJSON(cls, data, fields):
        return cls.fromdict(json.loads(data), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key in data and vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
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
        return cls.fromlist(list(csv.reader([data]))[0], fields)

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        try:
            from nltk.tree import Tree
        except ImportError:
            print('''Please install NLTK:
    $ pip install nltk''')
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [cls.fromlist(
                [t.leaves(), t.label()], fields) for t in tree.subtrees()]
        return cls.fromlist([tree.leaves(), tree.label()], fields)


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key: The key to use for sorting examples from this dataset in
            order to batch together examples with similar lengths and minimize
            padding.
        examples: The list of Examples in the dataset.
        fields: A dictionary containing the name of each column together with
            its corresponding Field object. Two columns with the same Field
            object will share a vocabulary.
    """

    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of examples and fields.

        Arguments:
            examples: List of Examples.
            fields: List of tuples of (name, field).
            filter_pred: Use only examples for which filter_pred(ex) is True,
                or use all examples if None. Default: None.
        """
        if filter_pred is not None:
            examples = list(filter(filter_pred, examples))
        self.examples = examples

        self.fields = dict(fields)

    @classmethod
    def splits(cls, path, train=None, validation=None, test=None, **kwargs):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            path: Common prefix of the splits' file paths.
            train: Suffix to add to path for the train set, or None for no
                train set. Default: None.
            validation: Suffix to add to path for the validation set, or None
                for no validation set. Default: None.
            test: Suffix to add to path for the test set, or None for no test
                set. Default: None.
            Remaining keyword arguments: Passed to the constructor of the
                dataset class being used.
        """
        train_data = None if train is None else cls(path + train, **kwargs)
        val_data = None if validation is None else cls(path + validation,
                                                       **kwargs)
        test_data = None if test is None else cls(path + test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


class ZipDataset(Dataset):
    """Defines a Dataset loaded from a downloadable zip archive.

    Attributes:
        url: URL where the zip archive can be downloaded.
        filename: Filename of the downloaded zip archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

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
        return os.path.join(path, '')


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path: Path to the data file.
            format: One of "CSV", "TSV", or "JSON" (case-insensitive).
            fields: For CSV and TSV formats, list of tuples of (name, field).
                The list should be in the same order as the columns in the CSV
                or TSV file, while tuples of (name, None) represent columns
                that will be ignored. For JSON format, dictionary whose keys
                are the JSON keys and whose values are tuples of (name, field).
                This allows the user to rename columns from their JSON key
                names or select a subset of columns to load while ignoring
                others not present in this dictionary.
        """
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromTSV, 'csv': Example.fromCSV}[format.lower()]

        with open(os.path.expanduser(path)) as f:
            examples = [make_example(line, fields) for line in f]

        if make_example in (Example.fromdict, Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
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
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    for p in batch(data, batch_size * 100):
        for b in shuffled(batch(sorted(p, key=key), batch_size)):
            yield b


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True, sort_field=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            if sort_field is not None:
                data.sort(key=lambda ex: -len(getattr(ex, sort_field)))
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
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch


class Iterator(object):
    """Defines an iterator that loads batches of data from a Dataset.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
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
                 train=True, repeat=None, shuffle=None, sort=None, sort_field=None):
        self.batch_size, self.train, self.dataset = batch_size, train, dataset
        self.iterations = 0
        self.repeat = train if repeat is None else repeat
        self.shuffle = train if shuffle is None else shuffle
        self.sort = not train if sort is None else sort
        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key
        if sort_field is None:
            self.sort_field = dataset.sort_field
        else:
            self.sort_field = sort_field
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
        self.batches = batch(self.data(), self.batch_size)
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
                            self.train, sort_field=self.sort_field)
            if not self.repeat:
                raise StopIteration


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def init_epoch(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key)
        if not self.repeat:
            self.iterations = 0


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
    """Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)
