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


def RawAG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.

    Create supervised learning dataset: RawAG_NEWS

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawAG_NEWS()
    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def RawSogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.

    Create supervised learning dataset: RawSogouNews

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawSogouNews()
    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def RawDBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.

    Create supervised learning dataset: RawDBpedia

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawDBpedia()
    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def RawYelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.

    Create supervised learning dataset: RawYelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawYelpReviewPolarity()
    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def RawYelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.

    Create supervised learning dataset: RawYelpReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawYelpReviewFull()
    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def RawYahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.

    Create supervised learning dataset: RawYahooAnswers

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawYahooAnswers()
    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def RawAmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.

    Create supervised learning dataset: RawAmazonReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawAmazonReviewPolarity()
    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def RawAmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.

    Create supervised learning dataset: RawAmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawAmazonReviewFull()
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


def RawIMDB(root='.data'):
    """ Defines RawIMDB datasets.

    Create supervised learning dataset: RawIMDB

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> train, test = torchtext.experimental.datasets.RawIMDB()
    """

    dataset_tar = download_from_url(URLS['IMDB'], root=root)
    extracted_files = extract_archive(dataset_tar)
    train_data = generate_imdb_data('train', extracted_files)
    test_data = generate_imdb_data('test', extracted_files)
    return (RawTextDataset(train_data),
            RawTextDataset(test_data))


DATASETS = {
    'RawAG_NEWS': RawAG_NEWS,
    'RawSogouNews': RawSogouNews,
    'RawDBpedia': RawDBpedia,
    'RawYelpReviewPolarity': RawYelpReviewPolarity,
    'RawYelpReviewFull': RawYelpReviewFull,
    'RawYahooAnswers': RawYahooAnswers,
    'RawAmazonReviewPolarity': RawAmazonReviewPolarity,
    'RawAmazonReviewFull': RawAmazonReviewFull,
    'RawIMDB': RawIMDB
}
