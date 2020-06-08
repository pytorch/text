import torch
import logging
import io
from torchtext.utils import download_from_url, extract_archive

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


class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, iterator, start=0, num_lines=None):
        """Initiate language modeling dataset.
        """
        super(RawTextIterableDataset, self).__init__()
        self._iterator = iterator
        self.has_setup = False
        self.start = start
        self.num_lines = num_lines

    def setup_iter(self, start=0, num_lines=None):
        self.start = start
        self.num_lines = num_lines
        self.has_setup = True

    def __iter__(self):
        if not self.has_setup:
            self.setup_iter()
        for i, item in enumerate(self._iterator):
            if i >= self.start:
                yield item
            if (self.num_lines is not None) and (i == (self.start + self.num_lines)):
                break

    def get_iterator(self):
        return self._iterator


def _setup_datasets(dataset_name, root='.data', data_select=('train', 'test', 'valid'), **kwargs):
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
        year = kwargs.get('year', 2010)
        language = kwargs.get('language', 'en')
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

    return tuple(RawTextIterableDataset(data[item]) for item in data_select)


def WikiText2(*args, **kwargs):
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

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
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

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
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

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)


def WMTNewsCrawl(*args, **kwargs):
    """ Defines WMT News Crawl.

    Create language modeling dataset: WMTNewsCrawl

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tuple for the returned datasets.
            (Default: 'train')
    """

    return _setup_datasets(*(("WMTNewsCrawl",) + args), **kwargs)


DATASETS = {
    'WikiText2': WikiText2,
    'WikiText103': WikiText103,
    'PennTreebank': PennTreebank,
    'WMTNewsCrawl': WMTNewsCrawl
}
