import os
import re
import logging
import torch
import csv
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import generate_ngrams
import random
from tqdm import tqdm

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
    with open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = text_normalize(row[1])
            tokens = generate_ngrams(tokens, ngrams)
            dictionary.update(tokens)
    word_dictionary = OrderedDict()
    for (token, frequency) in dictionary.most_common():
        word_dictionary[token] = len(word_dictionary)
    return word_dictionary

def _create_data(dictionary, data_path):
    data = []
    labels = []
    with open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            cls = int(row[0]) - 1
            tokens = text_normalize(row[1])
            tokens = generate_ngrams(tokens, 2)
            tokens = torch.tensor([dictionary.get(entry, dictionary['UNK']) for entry in tokens])
            labels.append(cls)
            data.append(tokens)
    return data, labels

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

    def __init__(self, root, ngrams):
        """Initiate text-classification dataset.

        Arguments:
            url: url of the online raw data files.
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1
        """

        def _apply_preprocessing_to_csv(csv_filepath, tgt_filepath, function):
            lines = []
            with open(src_filepath) as src_data, open(tgt_filepath, 'w') as new_data:
                reader = unicode_csv_reader(src_data)
                lines.append((row[0], generate_ngrams(text_normalize(row[1]), ngrams)))
            return lines

        super(TextClassificationDataset, self).__init__()

        dataset_name = self.__class__.__name__
        dataset_root = os.path.join(root, dataset_name + '_csv')
        dataset_tar = dataset_root + '.tar.gz'

        if not os.path.exists(dataset_tar):
            import pdb; pdb.set_trace()
            download_from_url(URLS[dataset_name], dataset_tar)
            logging.info('Dataset %s downloaded.' % dataset_name)

        extracted_files = extract_archive(dataset_tar, root)
        print(extracted_files)
        for fname in extracted_files:
            if fname.endswith('train.csv'):
                train_csv_path = fname
            if fname.endswith('test.csv'):
                test_csv_path = fname

        dictionary = _build_dictionary_from_path(train_csv_path, ngrams)
        dictionary['UNK'] = len(dictionary)

        self.train_data, self.train_labels = _create_data(dictionary, train_csv_path)
        self.test_data, self.test_labels = _create_data(dictionary, test_csv_path)
        self.dictionary = dictionary
        self.unk = 'UNK'


class AG_NEWS(TextClassificationDataset):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: AG_NEWS

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.AG_NEWS(ngrams=3)

        """

        super(AG_NEWS, self).__init__(root, ngrams)


class SogouNews(TextClassificationDataset):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: SogouNews

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.SogouNews(ngrams=3)

        """

        super(SogouNews, self).__init__(root, ngrams)


class DBpedia(TextClassificationDataset):
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
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: DBpedia

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.DBpedia(ngrams=3)

        """

        super(DBpedia, self).__init__(root, ngrams)


class YelpReviewPolarity(TextClassificationDataset):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: YelpReviewPolarity

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.YelpReviewPolarity(ngrams=3)

        """

        super(YelpReviewPolarity, self).__init__(
                                                 root, ngrams)


class YelpReviewFull(TextClassificationDataset):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: YelpReviewFull

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.YelpReviewFull(ngrams=3)

        """

        super(YelpReviewFull, self).__init__(
                                             root, ngrams)


class YahooAnswers(TextClassificationDataset):
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
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: YahooAnswers

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.YahooAnswers(ngrams=3)

        """

        super(YahooAnswers, self).__init__(
                                           root, ngrams)


class AmazonReviewPolarity(TextClassificationDataset):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: AmazonReviewPolarity

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

        """

        super(AmazonReviewPolarity, self).__init__(
                                                   root, ngrams)


class AmazonReviewFull(TextClassificationDataset):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)
     """
    def __init__(self, root='.data', ngrams=1):
        """Create supervised learning dataset: AmazonReviewFull

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.AmazonReviewFull(ngrams=3)

        """

        super(AmazonReviewFull, self).__init__(
                                               root, ngrams)
