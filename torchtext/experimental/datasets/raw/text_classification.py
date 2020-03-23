import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader

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


def _create_data_from_csv(data_path):
    data = []
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            data.append((int(row[0]), ' '.join(row[1:])))
    return data


class RawTextDataset(torch.utils.data.Dataset):
    """Defines an abstraction for raw text datasets.
    """

    def __init__(self, data):
        """Initiate text-classification dataset.
        """

        super(RawTextDataset, self).__init__()
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def _setup_datasets(dataset_name, root='.data'):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    train_data = _create_data_from_csv(train_csv_path)
    test_data = _create_data_from_csv(test_csv_path)
    return (RawTextDataset(train_data),
            RawTextDataset(test_data))


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.

    Create supervised learning dataset: AG_NEWS

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.AG_NEWS()
    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.

    Create supervised learning dataset: SogouNews

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.SogouNews()
    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def DBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.

    Create supervised learning dataset: DBpedia

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.DBpedia()
    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.

    Create supervised learning dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.YelpReviewPolarity()
    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.

    Create supervised learning dataset: YelpReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.YelpReviewFull()
    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.

    Create supervised learning dataset: YahooAnswers

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.YahooAnswers()
    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.

    Create supervised learning dataset: AmazonReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.AmazonReviewPolarity()
    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.

    Create supervised learning dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.AmazonReviewFull()
    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


def generate_imdb_data(key, extracted_files):
    data_set = []
    for fname in extracted_files:
        if 'urls' in fname:
            continue
        elif key in fname and ('pos' in fname or 'neg' in fname):
            with io.open(fname, encoding="utf8") as f:
                label = 1 if 'pos' in fname else 0
                data_set.append((label, f.read()))
    return data_set


def IMDB(root='.data'):
    """ Defines IMDB datasets.

    Create supervised learning dataset: IMDB

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.raw_datasets.IMDB()
    """

    dataset_tar = download_from_url(URLS['IMDB'], root=root)
    extracted_files = extract_archive(dataset_tar)
    train_data = generate_imdb_data('train', extracted_files)
    test_data = generate_imdb_data('test', extracted_files)
    return (TextDataset(train_data),
            TextDataset(test_data))


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
