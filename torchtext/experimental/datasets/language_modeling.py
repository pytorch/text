import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets.raw import language_modeling as raw
from torchtext.experimental.functional import vocab_func, totensor, sequential_transforms


def build_vocab(data, transforms):
    def apply_transforms(data):
        for line in data:
            tokens = transforms(line)
            if len(tokens) > 0:
                yield tokens
    return build_vocab_from_iterator(apply_transforms(data))


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank
             - WMTNewsCrawl

    """

    def __init__(self, data, vocab, transforms, single_line):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.
            transforms: Text string transforms.

        """

        super(LanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.transforms = transforms
        self.single_line = single_line
        if single_line:
            self.data = torch.cat(tuple(transforms(row) for row in data), axis=0)
        else:
            self.data = data

    def __getitem__(self, i):
        if self.single_line:
            return self.data[i]
        else:
            return self.transforms(self.data[i])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab


def _setup_datasets(dataset_name, tokenizer=None, root='.data', vocab=None,
                    data_select=('train', 'test', 'valid'), single_line=True):
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')
    text_transform = sequential_transforms(tokenizer)

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'valid', 'test'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))

    if not single_line and dataset_name != 'WikiText103':
        raise TypeError('single_line must be True except for WikiText103')
    if dataset_name == 'WMTNewsCrawl':
        train, = raw.DATASETS[dataset_name](root=root, data_select=('train',))
        if single_line:
            raw_data = {'train': [" ".join([txt for txt in train]), ]}
        else:
            raw_data = {'train': [txt for txt in train]}
    else:
        train, test, valid = raw.DATASETS[dataset_name](root=root, data_select=('train', 'test', 'valid'))
        # Cache raw text iterable dataset
        if single_line:
            raw_data = {'train': [" ".join([txt for txt in train]), ],
                        'valid': [" ".join(txt for txt in valid), ],
                        'test': [" ".join(txt for txt in test), ]}
        else:
            raw_data = {'train': [txt for txt in train],
                        'valid': [txt for txt in valid],
                        'test': [txt for txt in test]}

    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocab = build_vocab(raw_data['train'], text_transform)
    text_transform = sequential_transforms(text_transform, vocab_func(vocab),
                                           totensor(dtype=torch.long))
    return tuple(LanguageModelingDataset(raw_data[item], vocab, text_transform, single_line)
                 for item in data_select)


def WikiText2(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText2(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText103(tokenizer=tokenizer, vocab=vocab,
                                         data_select='valid')

    """

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.

    Examples:
        >>> from torchtext.experimental.datasets import PennTreebank
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = PennTreebank(tokenizer=tokenizer, vocab=vocab,
                                          data_select='valid')

    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)


def WMTNewsCrawl(*args, **kwargs):
    """ Defines WMTNewsCrawl datasets.

    Create language modeling dataset: WMTNewsCrawl
    returns the train set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train',))
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.
    Examples:
        >>> from torchtext.experimental.datasets import WMTNewsCrawl
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, = WMTNewsCrawl(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("WMTNewsCrawl",) + args), **kwargs)


DATASETS = {
    'WikiText2': WikiText2,
    'WikiText103': WikiText103,
    'PennTreebank': PennTreebank,
    'WMTNewsCrawl': WMTNewsCrawl
}
