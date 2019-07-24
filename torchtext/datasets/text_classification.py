import os
import re
import logging
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import generate_ngrams
from torchtext.vocab import build_dictionary
import random
from tqdm import tqdm

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

def _build_dictionary(dataset, field, data_name, **kwargs):
    """Construct the Vocab object for the field from a dataset.

    Arguments:
        dataset: Dataset with the iterable data.
        field: Field object with the information of the special tokens.
        data_name: The names of data used to build vocab (e.g. 'text', 'label').
            It must be the attributes of dataset's examples.
        Remaining keyword arguments: Passed to the constructor of Vocab.

    Examples:
        >>> field.vocab = build_vocab(dataset, field, 'text')
    """
    counter = Counter()
    for x in dataset:
        x = getattr(x, data_name)
        if not field.sequential:
            x = [x]
        try:
            counter.update(x)
        except TypeError:
            counter.update(chain.from_iterable(x.text))
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token] if tok is not None))
    return Vocab(counter, specials=specials, **kwargs)

def download_extract_archive(url, raw_folder, dataset_name):
    """Download the dataset if it doesn't exist in processed_folder already."""

    train_csv_path = os.path.join(raw_folder,
                                  dataset_name + '_csv',
                                  'train.csv')
    test_csv_path = os.path.join(raw_folder,
                                 dataset_name + '_csv',
                                 'test.csv')
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        return

    os.makedirs(raw_folder)
    filename = dataset_name + '_csv.tar.gz'
    url = url
    path = os.path.join(raw_folder, filename)
    download_from_url(url, path)
    extract_archive(path, raw_folder, remove_finished=True)

    logging.info('Dataset %s downloaded.' % dataset_name)


# def text_normalize(line):
#     """Normalize text string and separate label/text."""
#     line = line.lower()
#     label, text = line.split(",", 1)
#     label = "__label__" + re.sub(r'[^0-9\s]', '', label)
#     text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
#     text = ' '.join(text.split())
#     line = label + ' , ' + text + ' \n'
#     return line

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
        reader = csv.reader(f)
        for row in reader:
            tokens = text_normalize(row[1])
            tokens = generate_ngrams(tokens, ngrams)
            dictionary.update(tokens)
    word_dictionary = OrderedDict()
    for (token, frequency) in dictionary.most_common():
        word_dictionary[token] = len(word_dictionary)
    return word_dictionary

def _create_data(dictionary, data_path):
    all_data = []
    with open(data_path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            cls = int(row[0]) - 1
            tokens = text_normalize(row[1])
            tokens = generate_ngrams(tokens, 2)
            tokens = torch.tensor([dictionary.get(entry, dictionary['UNK']) for entry in tokens])
            all_data.append((cls, tokens))
    return all_data

# def _preprocess(raw_folder, processed_folder, dataset_name):
#     """Preprocess the csv files."""
# 
#     raw_folder = os.path.join(raw_folder, dataset_name.lower() + '_csv')
# 
#     if os.path.exists(processed_folder) is not True:
#         os.makedirs(processed_folder)
# 
# 
#     _apply_preprocessing_to_csv(
#         os.path.join(raw_folder, 'train.csv')
#         os.path.join(processed_folder, dataset_name + '.train')
#     )
# 
#     _apply_preprocessing_to_csv(
#         os.path.join(raw_folder, 'test.csv')
#         os.path.join(processed_folder, dataset_name + '.test')
#     )
#     logging.info("Dataset %s preprocessed." % dataset_name)


# def _load_text_classification_data(src_data, fields, ngrams=1):
#     """Load train/test data from a file and generate data
#         examples with ngrams.
#     """
# 
#     def label_text_processor(line, fields, ngrams=1):
#         """Process text string and generate examples for dataset."""
#         fields = [('text', fields['text']), ('label', fields['label'])]
#         label, text = line.split(",", 1)
#         label = float(label.split("__label__")[1])
#         ex = data.Example.fromlist([text, label], fields)
#         tokens = ex.text[1:]  # Skip the first space '\t'
#         ex.text = data.utils.generate_ngrams(tokens, ngrams)
#         return ex
# 
#     examples = []
#     for line in tqdm(src_data):
#         examples.append(label_text_processor(line, fields, ngrams))
#     return examples


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

    def __init__(self, root='.data', ngrams=1):
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
                reader = csv.reader(src_data)
                lines.append((row[0], generate_ngrams(text_normalize(row[1]), ngrams)))
            return lines

        super(TextClassificationDataset, self).__init__()

        self.dataset_name = self.__class__.__name__
        self.root = root
        self.raw_folder = os.path.join(root, self.__class__.__name__, 'raw')

        self.url = URLS[self.dataset_name]
        download_extract_archive(url, self.raw_folder, self.dataset_name)

        dictionary = _build_dictionary_from_path(os.path.join(raw_folder, 'train.csv'), ngrams)
        dictionary['UNK'] = len(self.dictionary)

        self.train_examples = _create_data(dictionary, os.path.join(raw_folder, 'train.csv'))
        self.test_examples = _create_data(dictionary, os.path.join(raw_folder, 'test.csv'))
        self.examples = self.train_examples + self.test_examples

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

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
            text_field: The field that will be used for the sentence. If not given,
                'spacy' token will be used.
            label_field: The field that will be used for the label. If not given,
                'float' token will be used.
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.AG_NEWS(ngrams=3)

        """
        self.url = URLS['AG_NEWS']
        super(AG_NEWS, self).__init__(self.url, root, text_field,
                                      label_field, ngrams)


class SogouNews(TextClassificationDataset):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology
     """
    def __init__(self, root='.data', text_field=None,
                 label_field=None, ngrams=1):
        """Create supervised learning dataset: SogouNews

        Arguments:
            root: Directory where the dataset are saved. Default: ".data"
            text_field: The field that will be used for the sentence. If not given,
                'spacy' token will be used.
            label_field: The field that will be used for the label. If not given,
                'float' token will be used.
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.SogouNews(ngrams=3)

        """
        self.url = URLS['SogouNews']
        super(SogouNews, self).__init__(self.url, root, text_field,
                                        label_field, ngrams)


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
            text_field: The field that will be used for the sentence. If not given,
                'spacy' token will be used.
            label_field: The field that will be used for the label. If not given,
                'float' token will be used.
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> text_cls = torchtext.datasets.DBpedia(ngrams=3)

        """
        self.url = URLS['DBpedia']
        super(DBpedia, self).__init__(self.url, root, ngrams)


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
        self.url = URLS['YelpReviewPolarity']
        super(YelpReviewPolarity, self).__init__(self.url,
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
        self.url = URLS['YelpReviewFull']
        super(YelpReviewFull, self).__init__(self.url,
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
        self.url = URLS['YahooAnswers']
        super(YahooAnswers, self).__init__(self.url,
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
        self.url = URLS['AmazonReviewPolarity']
        super(AmazonReviewPolarity, self).__init__(self.url,
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
        self.url = URLS['AmazonReviewFull']
        super(AmazonReviewFull, self).__init__(self.url,
                                               root, ngrams)
