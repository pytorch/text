import os
import re
import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import generate_ngrams

from collections import Counter
from collections import OrderedDict

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
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}


# TODO: Replicate below
#  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
#    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
#        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
#        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " "
_normalize_pattern_re = re.compile(r'[\W_]+')


def text_normalize(line):
    """
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - replacing all non-alphanumeric characters with whitespace
    Returns a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    line = _normalize_pattern_re.sub(' ', line)

    return line.split()


def _build_dictionary_from_path(data_path, ngrams):
    dictionary = Counter()
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        with tqdm(unit_scale=0, unit='lines') as t:
            for row in reader:
                tokens = text_normalize(row[1])
                tokens = generate_ngrams(tokens, ngrams)
                dictionary.update(tokens)
                t.update(1)
    word_dictionary = OrderedDict()
    for (token, frequency) in dictionary.most_common():
        word_dictionary[token] = len(word_dictionary)
    return word_dictionary


def _create_data(dictionary, data_path):
    data = []
    labels = []
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            cls = int(row[0]) - 1
            tokens = text_normalize(row[1])
            tokens = generate_ngrams(tokens, 2)
            tokens = torch.tensor(
                [dictionary.get(entry, dictionary['<unk>']) for entry in tokens])
            data.append((cls, tokens))
            labels.append(cls)
    return data, set(labels)


def _extract_data(root, dataset_name):
    dataset_root = os.path.join(root, dataset_name + '_csv')
    dataset_tar = dataset_root + '.tar.gz'

    if os.path.exists(dataset_tar):
        logging.info('Dataset %s already downloaded.' % dataset_name)
    else:
        download_from_url(URLS[dataset_name], dataset_tar)
        logging.info('Dataset %s downloaded.' % dataset_name)
    extracted_files = extract_archive(dataset_tar, root)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname
    return train_csv_path, test_csv_path


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

    def __init__(self, dictionary, data, labels):
        """Initiate text-classification dataset.

        Arguments:
            url: url of the online raw data files.
            root: Directory where the dataset are saved.
            ngrams: a contiguous sequence of n items from s string text.
        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._dictionary = dictionary

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_dictionary(self):
        return self._dictionary


def _setup_datasets(root, ngrams, dataset_name):
    train_csv_path, test_csv_path = _extract_data(root, dataset_name)

    logging.info('Building dictionary based on {}'.format(train_csv_path))
    # TODO: Clean up and use Vocab object
    # Standardized on torchtext.Vocab
    UNK = '<unk>'
    dictionary = _build_dictionary_from_path(train_csv_path, ngrams)
    dictionary[UNK] = len(dictionary)
    logging.info('Creating training data')
    train_data, train_labels = _create_data(dictionary, train_csv_path)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data(dictionary, test_csv_path)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(dictionary, train_data, train_labels),
            TextClassificationDataset(dictionary, test_data, test_labels))


def AG_NEWS(root='.data', ngrams=1):
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

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "AG_NEWS")


def SogouNews(root='.data', ngrams=1):
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

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "SogouNews")


def DBpedia(root='.data', ngrams=1):
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

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "DBpedia")


def YelpReviewPolarity(root='.data', ngrams=1):
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

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "YelpReviewPolarity")


def YelpReviewFull(root='.data', ngrams=1):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).

    Create supervised learning dataset: YelpReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "YelpReviewFull")


def YahooAnswers(root='.data', ngrams=1):
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

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YahooAnswers(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "YahooAnswers")


def AmazonReviewPolarity(root='.data', ngrams=1):
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

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "AmazonReviewPolarity")


def AmazonReviewFull(root='.data', ngrams=1):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)

    Create supervised learning dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the dataset are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(ngrams=3)

    """

    return _setup_datasets(root, ngrams, "AmazonReviewFull")


DATASETS = {
    'AG_NEWS': AG_NEWS,
    'SogouNews': SogouNews,
    'DBpedia': DBpedia,
    'YelpReviewPolarity': YelpReviewPolarity,
    'YelpReviewFull': YelpReviewFull,
    'YahooAnswers': YahooAnswers,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'AmazonReviewFull': AmazonReviewFull
}
