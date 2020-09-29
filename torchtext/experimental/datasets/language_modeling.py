import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets.raw import language_modeling as raw
from torchtext.experimental.datasets.raw.common import check_default_set


def build_vocab(data, transforms):
    def apply_transforms(data):
        for line in data:
            tokens = transforms(line)
            yield tokens
    return build_vocab_from_iterator(apply_transforms(data), len(data))


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
        self.data = data
        if single_line:
            self.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, self.data)))

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

    data_select = check_default_set(data_select, target_select=('train', 'test', 'valid'))

    if not single_line and dataset_name != 'WikiText103':
        raise TypeError('single_line must be True except for WikiText103')
    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        raw_train, = raw.DATASETS[dataset_name](root=root, data_select=('train',))
        vocab = build_vocab(raw_train, tokenizer)

    def text_transform(line):
        return torch.tensor([vocab[token] for token in tokenizer(line)], dtype=torch.long)

    raw_data = {}
    for name in data_select:
        raw_data[name], = raw.DATASETS[dataset_name](root=root, data_select=name)
        raw_data[name] = [text_transform(txt) for txt in raw_data[name]]

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
