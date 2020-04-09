import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets.raw import AG_NEWS as RawAG_NEWS
from torchtext.experimental.datasets.raw import SogouNews as RawSogouNews
from torchtext.experimental.datasets.raw import DBpedia as RawDBpedia
from torchtext.experimental.datasets.raw import YelpReviewPolarity as \
    RawYelpReviewPolarity
from torchtext.experimental.datasets.raw import YelpReviewFull as RawYelpReviewFull
from torchtext.experimental.datasets.raw import YahooAnswers as RawYahooAnswers
from torchtext.experimental.datasets.raw import AmazonReviewPolarity as \
    RawAmazonReviewPolarity
from torchtext.experimental.datasets.raw import AmazonReviewFull as RawAmazonReviewFull
from torchtext.experimental.datasets.raw import IMDB as RawIMDB


def vocab_func(vocab):
    def _forward(tok_iter):
        return [vocab[tok] for tok in tok_iter]
    return _forward


def totensor(dtype):
    def _forward(ids_list):
        return torch.tensor(ids_list).to(dtype)
    return _forward


def ngrams_func(ngrams):
    def _forward(token_list):
        _token_list = []
        for _i in range(ngrams + 1):
            _token_list += zip(*[token_list[i:] for i in range(_i)])
        return [' '.join(x) for x in _token_list]
    return _forward


def build_vocab(data, transforms):
    tok_list = []
    for (label, txt) in data:
        tok_list.append(transforms(txt))
    return build_vocab_from_iterator(tok_list)


def squential_transforms(*transforms):
    def _forward(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return _forward


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull
    """

    def __init__(self, data, vocab, transforms):
        """Initiate text-classification dataset.

        Arguments:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms  # (label_transforms, tokens_transforms)

    def __getitem__(self, i):
        label = self.data[i][0]
        txt = self.data[i][1]
        return (self.transforms[0](label), self.transforms[1](txt))

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labels = []
        for item in self.data:
            label = item[0]
            labels.apppend(self.transforms[0](label))
        return set(labels)

    def get_vocab(self):
        return self.vocab


def _setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None,
                    tokenizer=None, data_select=('train', 'test')):
    text_transform = []
    if not tokenizer:
        tokenizer = get_tokenizer('basic_english')
    text_transform = squential_transforms(tokenizer, ngrams_func(ngrams))

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))
    train, test = DATASETS[dataset_name](root=root)
    # Cache raw text iterable dataset
    raw_data = {'train': [(label, txt) for (label, txt) in train],
                'test': [(label, txt) for (label, txt) in test]}

    if not vocab:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocab = build_vocab(raw_data['train'], text_transform)
    text_transform = squential_transforms(text_transform, vocab_func(vocab),
                                          totensor(dtype=torch.long))
    label_transform = squential_transforms(totensor(dtype=torch.long))
    return tuple(TextClassificationDataset(raw_data[item], vocab,
                                           (label_transform, text_transform))
                 for item in data_select)


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech

    Create text classification dataset: AG_NEWS

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import AG_NEWS
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = AG_NEWS(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = AG_NEWS(tokenizer=tokenizer)
        >>> train, = AG_NEWS(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(('AG_NEWS',) + args), **kwargs)


def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology

    Create text classification dataset: SogouNews

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import SogouNews
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = SogouNews(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = SogouNews(tokenizer=tokenizer)
        >>> train, = SogouNews(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def DBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.
        The labels includes:
            - 1 : Company
            - 2 : EducationalInstitution
            - 3 : Artist
            - 4 : Athlete
            - 5 : OfficeHolder
            - 6 : MeanOfTransportation
            - 7 : Building
            - 8 : NaturalPlace
            - 9 : Village
            - 10 : Animal
            - 11 : Plant
            - 12 : Album
            - 13 : Film
            - 14 : WrittenWork

    Create text classification dataset: DBpedia

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import DBpedia
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = DBpedia(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = DBpedia(tokenizer=tokenizer)
        >>> train, = DBpedia(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create text classification dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import YelpReviewPolarity
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = YelpReviewPolarity(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = YelpReviewPolarity(tokenizer=tokenizer)
        >>> train, = YelpReviewPolarity(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).

    Create text classification dataset: YelpReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import YelpReviewFull
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = YelpReviewFull(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = YelpReviewFull(tokenizer=tokenizer)
        >>> train, = YelpReviewFull(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.
        The labels includes:
            - 1 : Society & Culture
            - 2 : Science & Mathematics
            - 3 : Health
            - 4 : Education & Reference
            - 5 : Computers & Internet
            - 6 : Sports
            - 7 : Business & Finance
            - 8 : Entertainment & Music
            - 9 : Family & Relationships
            - 10 : Politics & Government

    Create text classification dataset: YahooAnswers

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import YahooAnswers
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = YahooAnswers(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = YahooAnswers(tokenizer=tokenizer)
        >>> train, = YahooAnswers(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity

    Create text classification dataset: AmazonReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import AmazonReviewPolarity
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = AmazonReviewPolarity(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = AmazonReviewPolarity(tokenizer=tokenizer)
        >>> train, = AmazonReviewPolarity(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)

    Create text classification dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import AmazonReviewFull
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = AmazonReviewFull(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = AmazonReviewFull(tokenizer=tokenizer)
        >>> train, = AmazonReviewFull(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


def IMDB(*args, **kwargs):
    """ Defines IMDB datasets.
        The labels includes:
            - 0 : Negative
            - 1 : Positive

    Create sentiment analysis dataset: IMDB

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import IMDB
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = IMDB(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = IMDB(tokenizer=tokenizer)
        >>> train, = IMDB(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("IMDB",) + args), **kwargs)


DATASETS = {
    'AG_NEWS': RawAG_NEWS,
    'SogouNews': RawSogouNews,
    'DBpedia': RawDBpedia,
    'YelpReviewPolarity': RawYelpReviewPolarity,
    'YelpReviewFull': RawYelpReviewFull,
    'YahooAnswers': RawYahooAnswers,
    'AmazonReviewPolarity': RawAmazonReviewPolarity,
    'AmazonReviewFull': RawAmazonReviewFull,
    'IMDB': RawIMDB
}


LABELS = {
    'AG_NEWS': {1: 'World',
                2: 'Sports',
                3: 'Business',
                4: 'Sci/Tech'},
    'SogouNews': {1: 'Sports',
                  2: 'Finance',
                  3: 'Entertainment',
                  4: 'Automobile',
                  5: 'Technology'},
    'DBpedia': {1: 'Company',
                2: 'EducationalInstitution',
                3: 'Artist',
                4: 'Athlete',
                5: 'OfficeHolder',
                6: 'MeanOfTransportation',
                7: 'Building',
                8: 'NaturalPlace',
                9: 'Village',
                10: 'Animal',
                11: 'Plant',
                12: 'Album',
                13: 'Film',
                14: 'WrittenWork'},
    'YelpReviewPolarity': {1: 'Negative polarity',
                           2: 'Positive polarity'},
    'YelpReviewFull': {1: 'score 1',
                       2: 'score 2',
                       3: 'score 3',
                       4: 'score 4',
                       5: 'score 5'},
    'YahooAnswers': {1: 'Society & Culture',
                     2: 'Science & Mathematics',
                     3: 'Health',
                     4: 'Education & Reference',
                     5: 'Computers & Internet',
                     6: 'Sports',
                     7: 'Business & Finance',
                     8: 'Entertainment & Music',
                     9: 'Family & Relationships',
                     10: 'Politics & Government'},
    'AmazonReviewPolarity': {1: 'Negative polarity',
                             2: 'Positive polarity'},
    'AmazonReviewFull': {1: 'score 1',
                         2: 'score 2',
                         3: 'score 3',
                         4: 'score 4',
                         5: 'score 5'},
    'IMDB': {0: 'Negative',
             1: 'Positive'}
}
