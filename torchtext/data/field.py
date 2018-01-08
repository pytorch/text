# coding: utf8
from collections import Counter, OrderedDict
import six
import torch
from torch.autograd import Variable
from tqdm import tqdm

from .dataset import Dataset
from .pipeline import Pipeline
from .utils import get_tokenizer
from ..vocab import Vocab, SubwordVocab


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance, a text
    classification dataset contains sentences and their classes, while a
    machine translation dataset contains paired examples of text in two
    languages. Each of these types of data is represented by an RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return batch


class Field(RawField):
    """Defines a datatype together with instructions for converting to Tensor.

    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
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
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool).
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor types to the appropriate Python
    # numeric type.
    tensor_types = {
        torch.FloatTensor: float,
        torch.cuda.FloatTensor: float,
        torch.DoubleTensor: float,
        torch.cuda.DoubleTensor: float,
        torch.HalfTensor: float,
        torch.cuda.HalfTensor: float,

        torch.ByteTensor: int,
        torch.cuda.ByteTensor: int,
        torch.CharTensor: int,
        torch.cuda.CharTensor: int,
        torch.ShortTensor: int,
        torch.cuda.ShortTensor: int,
        torch.IntTensor: int,
        torch.cuda.IntTensor: int,
        torch.LongTensor: int,
        torch.cuda.LongTensor: int
    }

    def __init__(self, sequential=True, use_vocab=True, init_token=None, eos_token=None, fix_length=None,
                 tensor_type=torch.LongTensor, preprocessing=None, postprocessing=None, lower=False,
                 tokenize=(lambda s: s.split()), include_lengths=False, batch_first=False, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.tensor_type = tensor_type
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first

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
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device, train):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device, train=train)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
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
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
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
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)
        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None, train)

        arr = self.tensor_type(arr)
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


class ReversibleField(Field):

    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is list:
            self.use_revtok = False
        else:
            self.use_revtok = True
        if kwargs.get('tokenize') is None:
            kwargs['tokenize'] = 'revtok'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' UNK '
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch):
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if self.use_revtok:
            return [revtok.detokenize(ex) for ex in batch]
        return [''.join(ex) for ex in batch]


class SubwordField(ReversibleField):

    vocab_cls = SubwordVocab

    def __init__(self, **kwargs):
        kwargs['tokenize'] = 'subword'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = '�'
        super(SubwordField, self).__init__(**kwargs)

    def segment(self, *args):
        """Segment one or more datasets with this subword field.

        Arguments:
            Positional arguments: Dataset objects or other indexable
                mutable sequences to segment. If a Dataset object is provided,
                all columns corresponding to this field are used; individual
                columns can also be provided directly.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in tqdm(data, 'segmenting'):
                x[:] = self.vocab.segment(x)


class NestedField(Field):
    """A nested field.

    A nested field holds another field (called *nesting field*), accepts an untokenized
    string or a list string tokens and groups and treats them as one field as described
    by the nesting field. Every token will be preprocessed, padded, etc. in the manner
    specified by the nesting field. Note that this means a nested field always has
    ``sequential=True``. The two fields' vocabularies will be shared. Their
    numericalization results will be stacked into a single tensor. This field is
    primarily used to implement character embeddings. See ``tests/data/test_field.py``
    for examples on how to use this field.

    Arguments:
        nesting_field (Field): A field contained in this nested field.
        use_vocab (bool): Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: ``True``.
        init_token (str): A token that will be prepended to every example using this
            field, or None for no initial token. Default: ``None``.
        eos_token (str): A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: ``None``.
        fix_length (int): A fixed length that all examples using this field will be
            padded to, or ``None`` for flexible sequence lengths. Default: ``None``.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: ``torch.LongTensor``.
        preprocessing (Pipeline): The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: ``None``.
        postprocessing (Pipeline): A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool). Default: ``None``.
        tokenize (callable or str): The function used to tokenize strings using this
            field into sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: ``lambda s: s.split()``
        pad_token (str): The string token used as padding. If ``nesting_field`` is
            sequential, this will be set to its ``pad_token``. Default: ``"<pad>"``.
        pad_first (bool): Do the padding of the sequence at the beginning. Default:
            ``False``.
    """
    def __init__(self, nesting_field, use_vocab=True, init_token=None, eos_token=None,
                 fix_length=None, tensor_type=torch.LongTensor, preprocessing=None,
                 postprocessing=None, tokenize=lambda s: s.split(), pad_token='<pad>',
                 pad_first=False):
        if nesting_field.include_lengths:
            raise ValueError('nesting field cannot have include_lengths=True')

        if nesting_field.sequential:
            pad_token = nesting_field.pad_token
        super(NestedField, self).__init__(
            use_vocab=use_vocab,
            init_token=init_token,
            eos_token=eos_token,
            fix_length=fix_length,
            tensor_type=tensor_type,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            lower=nesting_field.lower,
            tokenize=tokenize,
            batch_first=True,
            pad_token=pad_token,
            unk_token=nesting_field.unk_token,
            pad_first=pad_first,
        )
        self.nesting_field = nesting_field

    def preprocess(self, xs):
        """Preprocess a single example.

        Firstly, tokenization and the supplied preprocessing pipeline is applied. Since
        this field is always sequential, the result is a list. Then, each element of
        the list is preprocessed using ``self.nesting_field.preprocess`` and the resulting
        list is returned.

        Arguments:
            xs (list or str): The input to preprocess.

        Returns:
            list: The preprocessed list.
        """
        return [self.nesting_field.preprocess(x)
                for x in super(NestedField, self).preprocess(xs)]

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        If ``self.nesting_field.sequential`` is ``False``, each example in the batch must
        be a list of string tokens, and pads them as if by a ``Field`` with
        ``sequential=True``. Otherwise, each example must be a list of list of tokens.
        Using ``self.nesting_field``, pads the list of tokens to
        ``self.nesting_field.fix_length`` if provided, or otherwise to the length of the
        longest list of tokens in the batch. Next, using this field, pads the result by
        filling short examples with ``self.nesting_field.pad_token``.

        Example:
            >>> import pprint
            >>> pp = pprint.PrettyPrinter(indent=4)
            >>>
            >>> nesting_field = Field(pad_token='<c>', init_token='<w>', eos_token='</w>')
            >>> field = NestedField(nesting_field, init_token='<s>', eos_token='</s>')
            >>> minibatch = [
            ...     [list('john'), list('loves'), list('mary')],
            ...     [list('mary'), list('cries')],
            ... ]
            >>> padded = field.pad(minibatch)
            >>> pp.pprint(padded)
            [   [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'j', 'o', 'h', 'n', '</w>', '<c>'],
                    ['<w>', 'l', 'o', 'v', 'e', 's', '</w>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>']],
                [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', 'c', 'r', 'i', 'e', 's', '</w>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>', '<c>', '<c>']]]

        Arguments:
            minibatch (list): Each element is a list of string if
                ``self.nesting_field.sequential`` is ``False``, a list of list of string
                otherwise.

        Returns:
            list: The padded minibatch.
        """
        minibatch = list(minibatch)
        if not self.nesting_field.sequential:
            return super(NestedField, self).pad(minibatch)

        # Save values of attributes to be monkeypatched
        old_pad_token = self.pad_token
        old_init_token = self.init_token
        old_eos_token = self.eos_token
        old_fix_len = self.nesting_field.fix_length
        # Monkeypatch the attributes
        if self.nesting_field.fix_length is None:
            max_len = max(len(xs) for ex in minibatch for xs in ex)
            fix_len = max_len + 2 - (self.nesting_field.init_token,
                                     self.nesting_field.eos_token).count(None)
            self.nesting_field.fix_length = fix_len
        self.pad_token = [self.pad_token] * self.nesting_field.fix_length
        if self.init_token is not None:
            self.init_token = self.nesting_field.pad([[self.init_token]])[0]
        if self.eos_token is not None:
            self.eos_token = self.nesting_field.pad([[self.eos_token]])[0]
        # Do padding
        padded = [self.nesting_field.pad(ex) for ex in minibatch]
        padded = super(NestedField, self).pad(padded)
        # Restore monkeypatched attributes
        self.nesting_field.fix_length = old_fix_len
        self.pad_token = old_pad_token
        self.init_token = old_init_token
        self.eos_token = old_eos_token

        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for nesting field and combine it with this field's vocab.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for the nesting field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources.extend(
                    [getattr(arg, name) for name, field in arg.fields.items()
                     if field is self]
                )
            else:
                sources.append(arg)

        flattened = []
        for source in sources:
            flattened.extend(source)
        self.nesting_field.build_vocab(*flattened, **kwargs)
        super(NestedField, self).build_vocab()
        self.vocab.extend(self.nesting_field.vocab)
        self.nesting_field.vocab = self.vocab

    def numericalize(self, arrs, device=None, train=True):
        """Convert a padded minibatch into a variable tensor.

        Each item in the minibatch will be numericalized independently and the resulting
        tensors will be stacked at the first dimension.

        Arguments:
            arr (List[List[str]]): List of tokenized and padded examples.
            device (int): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (bool): Whether the batch is for a training set. If False, the Variable
                will be created with volatile=True. Default: True.
        """
        numericalized = []
        for arr in arrs:
            numericalized_ex = self.nesting_field.numericalize(
                arr, device=device, train=train)
            numericalized.append(numericalized_ex)
        return torch.stack(numericalized)
