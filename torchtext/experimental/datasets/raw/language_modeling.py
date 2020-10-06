import logging
import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import check_default_set

URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'WMTNewsCrawl': 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'
}


def _setup_datasets(dataset_name, root, data_select, year, language):
    data_select = check_default_set(data_select, ('train', 'test', 'valid'))
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')

    if dataset_name == 'PennTreebank':
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        extracted_files = [download_from_url(URLS['PennTreebank'][select_to_index[key]],
                                             root=root) for key in data_select]
    elif dataset_name == 'WMTNewsCrawl':
        if not (data_select == ['train'] or set(data_select).issubset(set(('train',)))):
            raise ValueError("WMTNewsCrawl only creates a training dataset. "
                             "data_select should be 'train' "
                             "or ('train',), got {}.".format(data_select))
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)
        file_name = 'news.{}.{}.shuffled'.format(year, language)
        extracted_files = [f for f in extracted_files if file_name in f]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    _path = {}
    for item in data_select:
        for fname in extracted_files:
            if item in fname:
                _path[item] = fname

    data = {}
    for item in _path.keys():
        logging.info('Creating {} data'.format(item))
        data[item] = iter(io.open(_path[item], encoding="utf8"))

    return tuple(RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name], data[item]) for item in data_select)


def WikiText2(root='.data', data_select=('train', 'test', 'valid')):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.raw.datasets import WikiText2
        >>> train_dataset, test_dataset, valid_dataset = WikiText2()
        >>> valid_dataset, = WikiText2(data_select='valid')

    """

    return _setup_datasets("WikiText2", root, data_select, None, None)


def WikiText103(root='.data', data_select=('train', 'test', 'valid')):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: the returned datasets (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test').
            If 'train' is not in the tuple, an vocab object should be provided which will
            be used to process valid and/or test data.

    Examples:
        >>> from torchtext.experimental.datasets.raw import WikiText103
        >>> train_dataset, test_dataset, valid_dataset = WikiText103()
        >>> valid_dataset, = WikiText103(data_select='valid')
    """

    return _setup_datasets("WikiText103", root, data_select, None, None)


def PennTreebank(root='.data', data_select=('train', 'test', 'valid')):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets.raw import PennTreebank
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank()
        >>> valid_dataset, = PennTreebank(data_select='valid')

    """

    return _setup_datasets("PennTreebank", root, data_select, None, None)


def WMTNewsCrawl(root='.data', data_select=('train'), year=2010, language='en'):
    """ Defines WMT News Crawl.

    Create language modeling dataset: WMTNewsCrawl

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tuple for the returned datasets.
            (Default: 'train')
        year: the year of the dataset (Default: 2010)
        language: the language of the dataset (Default: 'en')

    Note: unlike other language modeling datasets, WMTNewsCrawl provides a single dataset based on the year and language. There
        is no concept of train/valid/test for this case.
    """

    return _setup_datasets("WMTNewsCrawl", root, data_select, year, language)


DATASETS = {
    'WikiText2': WikiText2,
    'WikiText103': WikiText103,
    'PennTreebank': PennTreebank,
    'WMTNewsCrawl': WMTNewsCrawl
}
NUM_LINES = {
    'WikiText2': 36718,
    'WikiText103': 1801350,
    'PennTreebank': 42068,
    'WMTNewsCrawl': 17676013,
}
