import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torchtext.data.functional import numericalize_tokens_from_iterator

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


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    for cls, tokens in iterator:
        if include_unk:
            tokens = torch.tensor([vocab[token] for token in tokens])
        else:
            token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                    for token in tokens]))
            tokens = torch.tensor(token_ids)
        if len(tokens) == 0:
            logging.info('Row contains no tokens.')
        data.append((cls, tokens))
        labels.append(cls)
    return data, set(labels)


def _imdb_iterator(key, extracted_files, tokenizer, ngrams, yield_cls=False):
    for fname in extracted_files:
        if 'urls' in fname:
            continue
        elif key in fname:
            tokens = []
            if 'pos' in fname:
                with io.open(fname, encoding="utf8") as f:
                    tokens += tokenizer(f.read())
                if yield_cls:
                    yield int(1), ngrams_iterator(tokens, ngrams)
                else:
                    yield ngrams_iterator(tokens, ngrams)
            elif 'neg' in fname:
                with io.open(fname, encoding="utf8") as f:
                    tokens += tokenizer(f.read())
                if yield_cls:
                    yield int(0), ngrams_iterator(tokens, ngrams)
                else:
                    yield ngrams_iterator(tokens, ngrams)


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


def _setup_datasets(dataset_name, root='.data', ngrams=2, vocab=None,
                    include_unk=False, tokenizer=get_tokenizer("basic_english")):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    if vocab is None:
        logging.info('Building Vocab based on train data')
        if dataset_name == 'IMDB':
            _data_iterator = _imdb_iterator('train', extracted_files, tokenizer, ngrams)
        else:
            _data_iterator = _csv_iterator(train_csv_path, tokenizer, ngrams)
        vocab = build_vocab_from_iterator(_data_iterator)
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))

    logging.info('Creating training data')
    if dataset_name == 'IMDB':
        _data_iterator = _imdb_iterator('train', extracted_files,
                                        tokenizer, ngrams, yield_cls=True)
    else:
        _data_iterator = _csv_iterator(train_csv_path, tokenizer, ngrams, yield_cls=True)
    train_data, train_labels = _create_data_from_iterator(
        vocab, _data_iterator, include_unk)

    logging.info('Creating testing data')
    if dataset_name == 'IMDB':
        _data_iterator = _imdb_iterator('test', extracted_files,
                                        tokenizer, ngrams, yield_cls=True)
    else:
        _data_iterator = _csv_iterator(test_csv_path, tokenizer, ngrams, yield_cls=True)
    test_data, test_labels = _create_data_from_iterator(
        vocab, _data_iterator, include_unk)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))


def AG_NEWS(*args, **kwargs):
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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)

    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def SogouNews(*args, **kwargs):
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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)

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

    Create supervised learning dataset: DBpedia

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=3)

    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(ngrams=3)

    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(ngrams=3)

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

    Create supervised learning dataset: YahooAnswers

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YahooAnswers(ngrams=3)

    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
       >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(ngrams=3)

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
        include_unk: include unknown token in the data (Default: False)
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.

    Examples:
        >>> from torchtext.datasets import IMDB
        >>> from torchtext.data.utils import get_tokenizer
        >>> train_dataset, test_dataset = IMDB(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset = IMDB(tokenizer=tokenizer)
    """

    return _setup_datasets(*(('IMDB',) + args), **kwargs)


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
