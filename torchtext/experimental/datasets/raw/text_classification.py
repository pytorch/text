import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import check_default_set
from torchtext.experimental.datasets.raw.common import wrap_datasets

URLS = {
    'AG_NEWS':
        {'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
         'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'},
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
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            yield int(row[0]), ' '.join(row[1:])


def _setup_datasets(dataset_name, root, split_, offset):
    split = check_default_set(split_, ('train', 'test'), dataset_name)
    if dataset_name == 'AG_NEWS':
        extracted_files = [download_from_url(URLS[dataset_name][item], root=root,
                                             hash_value=MD5['AG_NEWS'][item],
                                             hash_type='md5') for item in ('train', 'test')]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root,
                                        hash_value=MD5[dataset_name], hash_type='md5')
        extracted_files = extract_archive(dataset_tar)

    cvs_path = {}
    for fname in extracted_files:
        if fname.endswith('train.csv'):
            cvs_path['train'] = fname
        if fname.endswith('test.csv'):
            cvs_path['test'] = fname
    return wrap_datasets(tuple(RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name][item],
                                                      _create_data_from_csv(cvs_path[item]), offset=offset) for item in split), split_)


def AG_NEWS(root='.data', split=('train', 'test'), offset=0):
    """AG_NEWS dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.AG_NEWS()
    """

    return _setup_datasets("AG_NEWS", root, split, offset)


def SogouNews(root='.data', split=('train', 'test'), offset=0):
    """SogouNews dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.SogouNews()
    """

    return _setup_datasets("SogouNews", root, split, offset)


def DBpedia(root='.data', split=('train', 'test'), offset=0):
    """DBpedia dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.DBpedia()
    """

    return _setup_datasets("DBpedia", root, split, offset)


def YelpReviewPolarity(root='.data', split=('train', 'test'), offset=0):
    """YelpReviewPolarity dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.YelpReviewPolarity()
    """

    return _setup_datasets("YelpReviewPolarity", root, split, offset)


def YelpReviewFull(root='.data', split=('train', 'test'), offset=0):
    """YelpReviewFull dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.YelpReviewFull()
    """

    return _setup_datasets("YelpReviewFull", root, split, offset)


def YahooAnswers(root='.data', split=('train', 'test'), offset=0):
    """YahooAnswers dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.YahooAnswers()
    """

    return _setup_datasets("YahooAnswers", root, split, offset)


def AmazonReviewPolarity(root='.data', split=('train', 'test'), offset=0):
    """AmazonReviewPolarity dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.AmazonReviewPolarity()
    """

    return _setup_datasets("AmazonReviewPolarity", root, split, offset)


def AmazonReviewFull(root='.data', split=('train', 'test'), offset=0):
    """AmazonReviewFull dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.AmazonReviewFull()
    """

    return _setup_datasets("AmazonReviewFull", root, split, offset)


def generate_imdb_data(key, extracted_files):
    for fname in extracted_files:
        if 'urls' in fname:
            continue
        elif key in fname and ('pos' in fname or 'neg' in fname):
            with io.open(fname, encoding="utf8") as f:
                label = 'pos' if 'pos' in fname else 'neg'
                yield label, f.read()


def IMDB(root='.data', split=('train', 'test'), offset=0):
    """IMDB dataset

    Separately returns the raw training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'test')
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> train, test = torchtext.experimental.datasets.raw.IMDB()
    """
    split_ = check_default_set(split, ('train', 'test'), 'IMDB')
    dataset_tar = download_from_url(URLS['IMDB'], root=root,
                                    hash_value=MD5['IMDB'], hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    return wrap_datasets(tuple(RawTextIterableDataset("IMDB", NUM_LINES["IMDB"][item],
                                                      generate_imdb_data(item,
                                                                         extracted_files), offset=offset) for item in split_), split)


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
NUM_LINES = {
    'AG_NEWS': {'train': 120000, 'test': 7600},
    'SogouNews': {'train': 450000, 'test': 60000},
    'DBpedia': {'train': 560000, 'test': 70000},
    'YelpReviewPolarity': {'train': 560000, 'test': 38000},
    'YelpReviewFull': {'train': 650000, 'test': 50000},
    'YahooAnswers': {'train': 1400000, 'test': 60000},
    'AmazonReviewPolarity': {'train': 3600000, 'test': 400000},
    'AmazonReviewFull': {'train': 3000000, 'test': 650000},
    'IMDB': {'train': 25000, 'test': 25000}
}
MD5 = {
    'AG_NEWS': {'train': 'b1a00f826fdfbd249f79597b59e1dc12', 'test': 'd52ea96a97a2d943681189a97654912d'},
    'SogouNews': '0c1700ba70b73f964dd8de569d3fd03e',
    'DBpedia': 'dca7b1ae12b1091090db52aa7ec5ca64',
    'YelpReviewPolarity': '620c8ae4bd5a150b730f1ba9a7c6a4d3',
    'YelpReviewFull': 'f7ddfafed1033f68ec72b9267863af6c',
    'YahooAnswers': 'f3f9899b997a42beb24157e62e3eea8d',
    'AmazonReviewPolarity': 'fe39f8b653cada45afd5792e0f0e8f9b',
    'AmazonReviewFull': '57d28bd5d930e772930baddf36641c7c',
    'IMDB': '7c2ac02c03563afcf9b574c7e56c153a'
}
