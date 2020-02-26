import torch
import logging
import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import numericalize_tokens_from_iterator

URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'WMTNewsCrawl': 'http://www.statmt.org/wmt11/training-monolingual-news-{}.tgz'
}


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank
             - WMTNewsCrawl
    """

    def __init__(self, data, vocab):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)

        """

        super(LanguageModelingDataset, self).__init__()
        self.data = data
        self.vocab = vocab

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab


def _get_datafile_path(key, extracted_files):
    for fname in extracted_files:
        if key in fname:
            return fname


def _setup_datasets(dataset_name, tokenizer=get_tokenizer("basic_english"),
                    root='.data', vocab=None, removed_tokens=[],
                    data_select=('train', 'test', 'valid'), **kwargs):

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

        year = kwargs.get('year', 2011)
        download_url = URLS[dataset_name].format(year)

        language = kwargs.get('language', 'en')
        fname = 'news.{year}.{language}.shuffled'.format(year=year, language=language)

        dataset_tar = download_from_url(download_url, root=root)
        extracted_files = extract_archive(dataset_tar)
        extracted_files = [f for f in extracted_files if fname in f]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)

    if vocab is None:
        if 'train' not in _path.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on {}'.format(_path['train']))
        txt_iter = iter(tokenizer(row) for row in io.open(_path['train'],
                                                          encoding="utf8"))
        vocab = build_vocab_from_iterator(txt_iter)
        logging.info('Vocab has {} entries'.format(len(vocab)))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    data = {}
    for item in _path.keys():
        data[item] = []
        logging.info('Creating {} data'.format(item))
        txt_iter = iter(tokenizer(row) for row in io.open(_path[item],
                                                          encoding="utf8"))
        _iter = numericalize_tokens_from_iterator(
            vocab, txt_iter, removed_tokens)
        for tokens in _iter:
            data[item] += [token_id for token_id in tokens]

    for key in data_select:
        if data[key] == []:
            raise TypeError('Dataset {} is empty!'.format(key))

    return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab)
                 for d in data_select)


def WikiText2(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText2(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: the returned datasets (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test').
            If 'train' is not in the tuple, an vocab object should be provided which will
            be used to process valid and/or test data.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText103(tokenizer=tokenizer, vocab=vocab,
                                         data_select='valid')

    """

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import PennTreebank
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = PennTreebank(tokenizer=tokenizer, vocab=vocab,
                                          data_select='valid')

    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)


def WMTNewsCrawl(*args, **kwargs):
    """ Defines WMT News Crawl.
        Create language modeling dataset: WMTNewsCrawl
        Creates a training set only
        Arguments:
            tokenizer: the tokenizer used to preprocess raw text data.
                The default one is basic_english tokenizer in fastText. spacy tokenizer
                is supported as well (see example below). A custom tokenizer is callable
                function with input of a string and output of a token list.
            root: Directory where the datasets are saved. Default: ".data"
            vocab: Vocabulary used for dataset. If None, it will generate a new
                vocabulary based on the train data set.
            removed_tokens: removed tokens from output dataset (Default: [])
            data_select: a string or tuple for the returned datasets.
                (Default: 'train')
                Only training dataset for News Crawl Corpus datasets by year.
            year: year of dataset to use. Choices are 2007-2011.
                See details below for memory and token size.
                (Default: 2011)
            language: language for dataset. Choices are cs, de, en, es, fr.
                (Default: 'en')

        News Crawl Corpus Dataset Details from `Statistical Machine Translation`__:
            +------+--------+--------------------+------------+----------+
            | Year | Memory | len(train_dataset) | len(vocab) | lines    |
            +======+========+====================+============+==========+
            | 2007 | 1.1 GB | 338142548          | 573176     | 13984262 |
            +------+--------+--------------------+------------+----------+
            | 2008 | 3.4 GB |838296018           | 1091099    | 34737842 |
            +------+--------+--------------------+------------+----------+
            | 2009 | 3.7 GB | 1027839909         | 1236276    | 44041422 |
            +------+--------+--------------------+------------+----------+
            | 2010 | 1.4 GB | 399857558          |  680765    | 17676013 |
            +------+--------+--------------------+------------+----------+
            | 2011 | 229 MB | 54831406           | 208279     | 2466169  |
            +------+--------+--------------------+------------+----------+

            NOTE: The memory refers to size of the tar.gz file downloaded.
            NOTE: The other metrics are for the english datasets.

        Examples:
            >>> from torchtext.experimental.datasets import WMTNewsCrawl
            >>> train_dataset, = WMTNewsCrawl(data_select='train',
                                              language='en',
                                              year=2011)
            >>> vocab = train_dataset.get_vocab()

        __ http://www.statmt.org/wmt11/translation-task.html#download
        """
    return _setup_datasets(*(("WMTNewsCrawl",) + args), **kwargs)
