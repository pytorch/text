import logging
import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import check_default_set
from torchtext.experimental.datasets.raw.common import wrap_datasets

URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank': {
        'train': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
        'test': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
        'valid': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'
    },
    'WMTNewsCrawl': 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'
}


def _setup_datasets(dataset_name, root, split, year, language, offset):
    if dataset_name == 'WMTNewsCrawl':
        split = check_default_set(split, ('train'), dataset_name)
    else:
        split = check_default_set(split, ('train', 'test', 'valid'), dataset_name)

    if dataset_name == 'PennTreebank':
        extracted_files = [download_from_url(URLS['PennTreebank'][key],
                                             root=root, hash_value=MD5['PennTreebank'][key],
                                             hash_type='md5') for key in split]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root, hash_value=MD5[dataset_name], hash_type='md5')
        extracted_files = extract_archive(dataset_tar)

    if dataset_name == 'WMTNewsCrawl':
        file_name = 'news.{}.{}.shuffled'.format(year, language)
        extracted_files = [f for f in extracted_files if file_name in f]

    path = {}
    for item in split:
        for fname in extracted_files:
            if item in fname:
                path[item] = fname

    datasets = []
    for item in split:
        logging.info('Creating {} data'.format(item))
        datasets.append(RawTextIterableDataset(dataset_name,
                                               NUM_LINES[dataset_name][item], iter(io.open(path[item], encoding="utf8")), offset=offset))

    return wrap_datasets(tuple(datasets), split)


def WikiText2(root='.data', split=('train', 'valid', 'test'), offset=0):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets. Default: ('train', 'valid, 'test')
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> from torchtext.experimental.raw.datasets import WikiText2
        >>> train_dataset, valid_dataset, test_dataset = WikiText2()
        >>> valid_dataset, = WikiText2(split='valid')

    """

    return _setup_datasets("WikiText2", root, split, None, None, offset)


def WikiText103(root='.data', split=('train', 'valid', 'test'), offset=0):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: the returned datasets. Default: ('train', 'valid','test')
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test').
            If 'train' is not in the tuple, an vocab object should be provided which will
            be used to process valid and/or test data.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> from torchtext.experimental.datasets.raw import WikiText103
        >>> train_dataset, valid_dataset, test_dataset = WikiText103()
        >>> valid_dataset, = WikiText103(split='valid')
    """

    return _setup_datasets("WikiText103", root, split, None, None, offset)


def PennTreebank(root='.data', split=('train', 'valid', 'test'), offset=0):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets ('train', 'valid', 'test') are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        offset: the number of the starting line. Default: 0

    Examples:
        >>> from torchtext.experimental.datasets.raw import PennTreebank
        >>> train_dataset, valid_dataset, test_dataset = PennTreebank()
        >>> valid_dataset, = PennTreebank(split='valid')

    """

    return _setup_datasets("PennTreebank", root, split, None, None, offset)


def WMTNewsCrawl(root='.data', split=('train'), year=2010, language='en', offset=0):
    """ Defines WMT News Crawl.

    Create language modeling dataset: WMTNewsCrawl

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets.
            (Default: 'train')
        year: the year of the dataset (Default: 2010)
        language: the language of the dataset (Default: 'en')
        offset: the number of the starting line. Default: 0

    Note: WMTNewsCrawl provides datasets based on the year and language instead of train/valid/test.
    """

    return _setup_datasets("WMTNewsCrawl", root, split, year, language, offset)


DATASETS = {
    'WikiText2': WikiText2,
    'WikiText103': WikiText103,
    'PennTreebank': PennTreebank,
    'WMTNewsCrawl': WMTNewsCrawl
}
NUM_LINES = {
    'WikiText2': {'train': 36718, 'valid': 3760, 'test': 4358},
    'WikiText103': {'train': 1801350, 'valid': 3760, 'test': 4358},
    'PennTreebank': {'train': 42068, 'valid': 3370, 'test': 3761},
    'WMTNewsCrawl': {'train': 17676013}
}
MD5 = {
    'WikiText2': '542ccefacc6c27f945fb54453812b3cd',
    'WikiText103': '9ddaacaf6af0710eda8c456decff7832',
    'PennTreebank': {'train': 'f26c4b92c5fdc7b3f8c7cdcb991d8420',
                     'valid': 'aa0affc06ff7c36e977d7cd49e3839bf',
                     'test': '8b80168b89c18661a38ef683c0dc3721'},
    'WMTNewsCrawl': '64150a352f3abe890a87f6c6838524a6'
}
