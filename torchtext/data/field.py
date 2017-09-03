from collections import Counter, OrderedDict
import six
import torch
from torch.autograd import Variable

from .dataset import Dataset
from .pipeline import Pipeline
from .utils import get_tokenizer
from ..vocab import Vocab


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
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
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
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
    """

    def __init__(
            self, sequential=True, use_vocab=True, init_token=None,
            eos_token=None, fix_length=None, tensor_type=torch.LongTensor,
            preprocessing=None, postprocessing=None, lower=False,
            tokenize=(lambda s: s.split()), include_lengths=False,
            batch_first=False, pad_token="<pad>"):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.tensor_type = tensor_type
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and not
                isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x)
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just returns the padded list.
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
                [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
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
        if isinstance(arr, tuple):
            arr, lengths = arr
        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)
        elif self.postprocessing is not None:
            arr = self.postprocessing(arr, train)
        arr = self.tensor_type(arr)
        if self.include_lengths:
            lengths = torch.LongTensor(lengths)
        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        if self.include_lengths:
            return Variable(arr, volatile=not train), lengths
        return Variable(arr, volatile=not train)
