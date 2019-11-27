import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab

URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA',
    'IMDB':
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
}


def _csv_iterator(data_path, tokenizer, ngrams, yield_cls=False):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def _create_data_from_iterator(vocab, iterator, removed_tokens):
    for cls, tokens in iterator:
        yield cls, iter(map(lambda x: vocab[x],
                        filter(lambda x: x not in removed_tokens, tokens)))


def _imdb_iterator(key, extracted_files, tokenizer, ngrams, yield_cls=False):
    for fname in extracted_files:
        if 'urls' in fname:
            continue
        elif key in fname and ('pos' in fname or 'neg' in fname):
            with io.open(fname, encoding="utf8") as f:
                label = 1 if 'pos' in fname else 0
                if yield_cls:
                    yield label, ngrams_iterator(tokenizer(f.read()), ngrams)
                else:
                    yield ngrams_iterator(tokenizer(f.read()), ngrams)


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

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.

        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}

        Examples:
            See the examples in examples/text_classification/

        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def generate_data_iterators(dataset_name, root, ngrams, tokenizer, data_select):
    if not tokenizer:
        tokenizer = get_tokenizer("basic_english")

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))

    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    path = {}
    for fname in extracted_files:
        if fname.endswith('train.csv') and 'train' in data_select:
            path['train'] = fname
        if fname.endswith('test.csv') and 'test' in data_select:
            path['test'] = fname

    iters_group = {}
    if 'train' in data_select:
        iters_group['vocab'] = _csv_iterator(path['train'], tokenizer, ngrams)
    for item in data_select:
        iters_group[item] = _csv_iterator(path[item], tokenizer,
                                          ngrams, yield_cls=True)
    return iters_group


def generate_imdb_data_iterators(dataset_name, root, ngrams, tokenizer, data_select):
    if not tokenizer:
        tokenizer = get_tokenizer("basic_english")

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))

    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    iters_group = {}
    if 'train' in data_select:
        iters_group['vocab'] = _imdb_iterator('train', extracted_files, tokenizer, ngrams)
    for item in data_select:
        iters_group[item] = _imdb_iterator(item, extracted_files,
                                           tokenizer, ngrams, yield_cls=True)
    return iters_group


def _setup_datasets(iters_group, vocab, removed_tokens):

    if vocab is None:
        if 'vocab' not in iters_group.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on train data')
        vocab = build_vocab_from_iterator(iters_group['vocab'])
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))

    data = {}
    for item in iters_group.keys():
        data[item] = {}
        data[item]['data'] = []
        data[item]['labels'] = []
        logging.info('Creating {} data'.format(item))
        data_iter = _create_data_from_iterator(vocab, iters_group[item], removed_tokens)
        for cls, tokens in data_iter:
            data[item]['data'].append((torch.tensor(cls),
                                       torch.tensor([token_id for token_id in tokens])))
            data[item]['labels'].append(cls)
        data[item]['labels'] = set(data[item]['labels'])

    data_select = list(iters_group.keys())
    data_select.remove('vocab') if 'vocab' in data_select else None
    return tuple(TextClassificationDataset(vocab, data[item]['data'],
                                           data[item]['labels']) for item in data_select)


def AG_NEWS(root='.data', ngrams=2, vocab=None, removed_tokens=[],
            tokenizer=None, data_select=('train', 'test')):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech

    Create supervised learning dataset: AG_NEWS

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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('AG_NEWS', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def SogouNews(root='.data', ngrams=2, vocab=None, removed_tokens=[],
              tokenizer=None, data_select=('train', 'test')):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology

    Create supervised learning dataset: SogouNews

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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('SogouNews', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def DBpedia(root='.data', ngrams=2, vocab=None, removed_tokens=[],
            tokenizer=None, data_select=('train', 'test')):
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

    Create supervised learning dataset: DBpedia

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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('DBpedia', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def YelpReviewPolarity(root='.data', ngrams=2, vocab=None, removed_tokens=[],
                       tokenizer=None, data_select=('train', 'test')):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create supervised learning dataset: YelpReviewPolarity

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
        data_select: a string or upel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('YelpReviewPolarity', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def YelpReviewFull(root='.data', ngrams=2, vocab=None, removed_tokens=[],
                   tokenizer=None, data_select=('train', 'test')):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).

    Create supervised learning dataset: YelpReviewFull

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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('YelpReviewFull', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def YahooAnswers(root='.data', ngrams=2, vocab=None, removed_tokens=[],
                 tokenizer=None, data_select=('train', 'test')):
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

    Create supervised learning dataset: YahooAnswers

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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YahooAnswers(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('YahooAnswers', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def AmazonReviewPolarity(root='.data', ngrams=2, vocab=None, removed_tokens=[],
                         tokenizer=None, data_select=('train', 'test')):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity

    Create supervised learning dataset: AmazonReviewPolarity

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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
       >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('AmazonReviewPolarity', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def AmazonReviewFull(root='.data', ngrams=2, vocab=None, removed_tokens=[],
                     tokenizer=None, data_select=('train', 'test')):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)

    Create supervised learning dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the dataset are saved. Default: ".data"
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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(ngrams=3)

    """

    return _setup_datasets(generate_data_iterators('AmazonReviewFull', root, ngrams,
                                                   tokenizer, data_select),
                           vocab, removed_tokens)


def IMDB(root='.data', ngrams=2, vocab=None, removed_tokens=[],
         tokenizer=None, data_select=('train', 'test')):
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
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.datasets import IMDB
        >>> from torchtext.data.utils import get_tokenizer
        >>> train_dataset, test_dataset = IMDB(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset = IMDB(tokenizer=tokenizer)
    """

    return _setup_datasets(generate_imdb_data_iterators('IMDB', root, ngrams,
                                                        tokenizer, data_select),
                           vocab, removed_tokens)


DATASETS = {
    'AG_NEWS': AG_NEWS,
    'SogouNews': SogouNews,
    'DBpedia': DBpedia,
    'YelpReviewPolarity': YelpReviewPolarity,
    'YelpReviewFull': YelpReviewFull,
    'YahooAnswers': YahooAnswers,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'AmazonReviewFull': AmazonReviewFull,
    'IMDB': IMDB
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
