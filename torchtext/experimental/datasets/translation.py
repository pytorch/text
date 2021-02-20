import torch
import logging
from torchtext.data.datasets_utils import check_default_set
from torchtext.data.datasets_utils import wrap_datasets
from torchtext import datasets as raw
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from ..functional import vocab_func, totensor, sequential_transforms

logger_ = logging.getLogger(__name__)


def build_vocab(data, transforms, index):
    def apply_transforms(data):
        for line in data:
            yield transforms(line[index])
    return build_vocab_from_iterator(apply_transforms(data), len(data))


def _setup_datasets(dataset_name,
                    train_filenames, valid_filenames, test_filenames,
                    split_, root, vocab, tokenizer):
    split = check_default_set(split_, ('train', 'valid', 'test'), dataset_name)
    src_vocab, tgt_vocab = vocab
    if tokenizer is None:
        src_tokenizer = get_tokenizer("spacy", language='de_core_news_sm')
        tgt_tokenizer = get_tokenizer("spacy", language='en_core_web_sm')
    elif isinstance(tokenizer, tuple):
        if len(tokenizer) == 2:
            src_tokenizer, tgt_tokenizer = tokenizer
        else:
            raise ValueError("tokenizer must have length of two for"
                             "source and target")
    else:
        raise ValueError(
            "tokenizer must be an instance of tuple with length two"
            "or None")

    if 'IWSLT' in dataset_name:
        src_language = train_filenames[0].split(".")[-1]
        tgt_language = train_filenames[1].split(".")[-1]
        if dataset_name == 'IWSLT2016':
            valid_set = valid_filenames[0].split(".")[2]
            test_set = test_filenames[0].split(".")[2]
            raw_datasets = raw.DATASETS[dataset_name](root=root,
                                                      split=split,
                                                      language_pair=(src_language, tgt_language),
                                                      valid_set=valid_set,
                                                      test_set=test_set)
        elif dataset_name == 'IWSLT2017':
            raw_datasets = raw.DATASETS[dataset_name](root=root,
                                                      split=split,
                                                      language_pair=(src_language, tgt_language))
        else:
            raise ValueError("{} is not supportd".format(dataset_name))

    else:
        raw_datasets = raw.DATASETS[dataset_name](train_filenames=train_filenames,
                                                  valid_filenames=valid_filenames,
                                                  test_filenames=test_filenames,
                                                  split=split, root=root)
    raw_data = {name: list(raw_dataset) for name, raw_dataset in zip(split, raw_datasets)}
    src_text_vocab_transform = sequential_transforms(src_tokenizer)
    tgt_text_vocab_transform = sequential_transforms(tgt_tokenizer)

    if src_vocab is None:
        if 'train' not in split:
            raise TypeError("Must pass a vocab if train is not selected.")
        logger_.info('Building src Vocab based on train data')
        src_vocab = build_vocab(raw_data["train"],
                                src_text_vocab_transform,
                                index=0)
    else:
        if not isinstance(src_vocab, Vocab):
            raise TypeError("Passed src vocabulary is not of type Vocab")
    logger_.info('src Vocab has %d entries', len(src_vocab))

    if tgt_vocab is None:
        if 'train' not in split:
            raise TypeError("Must pass a vocab if train is not selected.")
        logger_.info('Building tgt Vocab based on train data')
        tgt_vocab = build_vocab(raw_data["train"],
                                tgt_text_vocab_transform,
                                index=1)
    else:
        if not isinstance(tgt_vocab, Vocab):
            raise TypeError("Passed tgt vocabulary is not of type Vocab")
    logger_.info('tgt Vocab has %d entries', len(tgt_vocab))

    logger_.info('Building datasets for {}'.format(split))
    datasets = []
    for key in split:
        src_text_transform = sequential_transforms(src_text_vocab_transform,
                                                   vocab_func(src_vocab),
                                                   totensor(dtype=torch.long))
        tgt_text_transform = sequential_transforms(tgt_text_vocab_transform,
                                                   vocab_func(tgt_vocab),
                                                   totensor(dtype=torch.long))
        datasets.append(
            TranslationDataset(raw_data[key], (src_vocab, tgt_vocab),
                               (src_text_transform, tgt_text_transform)))

    return wrap_datasets(tuple(datasets), split_)


class TranslationDataset(torch.utils.data.Dataset):
    """Defines a dataset for translation.

    Currently, we only support the following datasets:

        - Multi30k
        - WMT14
        - IWSLT2016
        - IWSLT2017
    """

    def __init__(self, data, vocab, transforms):
        """Initiate translation dataset.

        Args:
            data: a tuple of source and target tensors, which include token ids
                numericalizing the string tokens.
                [(src_tensor0, tgt_tensor0), (src_tensor1, tgt_tensor1)]
            vocab: source and target Vocabulary object used for dataset.
                (src_vocab, tgt_vocab)
            transforms: a tuple of source and target string transforms.

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> src_data = torch.Tensor([token_id_s1, token_id_s2,
                                         token_id_s3, token_id_s1]).long()
            >>> tgt_data = torch.Tensor([token_id_t1, token_id_t2,
                                         token_id_t3, token_id_t1]).long()
            >>> src_vocab = build_vocab_from_iterator([['Übersetzungsdatensatz']])
            >>> tgt_vocab = build_vocab_from_iterator([['translation', 'dataset']])
            >>> dataset = TranslationDataset([(src_data, tgt_data)],
                                              (src_vocab, tgt_vocab))
        """

        super(TranslationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms

    def __getitem__(self, i):
        source = self.transforms[0](self.data[i][0])
        target = self.transforms[1](self.data[i][1])
        return (source, target)

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


def Multi30k(train_filenames=("train.de", "train.en"),
             valid_filenames=("val.de", "val.en"),
             test_filenames=("test_2016_flickr.de", "test_2016_flickr.en"),
             split=('train', 'valid', 'test'),
             root='.data',
             vocab=(None, None),
             tokenizer=None):
    """ Define translation datasets: Multi30k
    Separately returns train/valid/test datasets as a tuple

    Args:
        train_filenames: the source and target filenames for training.
            Default: ('train.de', 'train.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('val.de', 'val.en')
        test_filenames: the source and target filenames for test.
            Default: ('test2016.de', 'test2016.en')
        split: a string or tuple for the returned datasets, Default: ('train', 'valid', 'test')
            By default, all the three datasets (train, valid, test) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test data.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
            get_tokenizer("spacy", language='en_core_web_sm'))

        The available dataset include:

            test_2016_flickr.cs
            test_2016_flickr.de
            test_2016_flickr.en
            test_2016_flickr.fr
            test_2017_flickr.de
            test_2017_flickr.en
            test_2017_flickr.fr
            test_2017_mscoco.de
            test_2017_mscoco.en
            test_2017_mscoco.fr
            test_2018_flickr.en
            train.cs
            train.de
            train.en
            train.fr
            val.cs
            val.de
            val.en
            val.fr
            test_2016.1.de
            test_2016.1.en
            test_2016.2.de
            test_2016.2.en
            test_2016.3.de
            test_2016.3.en
            test_2016.4.de
            test_2016.4.en
            test_2016.5.de
            test_2016.5.en
            train.1.de
            train.1.en
            train.2.de
            train.2.en
            train.3.de
            train.3.en
            train.4.de
            train.4.en
            train.5.de
            train.5.en
            val.1.de
            val.1.en
            val.2.de
            val.2.en
            val.3.de
            val.3.en
            val.4.de
            val.4.en
            val.5.de
            val.5.en

    Examples:
        >>> from torchtext.experimental.datasets import Multi30k
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = (get_tokenizer("spacy", language='de'),
                         get_tokenizer("basic_english"))
        >>> train_dataset, valid_dataset, test_dataset = Multi30k(tokenizer=tokenizer)
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """

    return _setup_datasets("Multi30k", train_filenames, valid_filenames, test_filenames,
                           split, root, vocab, tokenizer)


def IWSLT2017(language_pair=('de', 'en'),
              split=('train', 'valid', 'test'),
              root='.data',
              vocab=(None, None),
              tokenizer=None):
    """ Define translation datasets: IWSLT2017
    Separately returns train/valid/test datasets
    The available datasets include following:
    - language pairs
    [('en', 'nl'), ('en', 'de'), ('en', 'it'), ('en', 'ro'), ('ro', 'de'),
    ('ro', 'en'), ('ro', 'nl'), ('ro', 'it'), ('de', 'ro'), ('de', 'en'),
    ('de', 'nl'), ('de', 'it'), ('it', 'en'), ('it', 'nl'), ('it', 'de'),
    ('it', 'ro'), ('nl', 'de'), ('nl', 'en'), ('nl', 'it'), ('nl', 'ro')]
    For additional details refer to source website below:
    https://wit3.fbk.eu/2017-01

    Args:
        language_pair: tuple or list of two elements: src and tgt language
        split: a string or tuple for the returned datasets, Default: ('train', 'valid', 'test')
            By default, all the three datasets (train, valid, test) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test data.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
            get_tokenizer("spacy", language='en_core_web_sm'))

    Examples:
        >>> from torchtext.experimental.datasets import IWSLT
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = IWSLT(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """
    year = 17
    valid_set = 'dev2010'
    test_set = 'tst2010'

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple")

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements'

    src_language, tgt_language = language_pair[0], language_pair[1]

    train_filenames = 'train.{}-{}.{}'.format(src_language, tgt_language, src_language), 'train.{}-{}.{}'.format(src_language, tgt_language, tgt_language)
    valid_filenames = 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, valid_set, src_language, tgt_language, src_language), 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, valid_set, src_language, tgt_language, tgt_language)
    test_filenames = 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, test_set, src_language, tgt_language, src_language), 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, test_set, src_language, tgt_language, tgt_language)

    return _setup_datasets("IWSLT2017", train_filenames, valid_filenames, test_filenames,
                           split, root, vocab, tokenizer)


def IWSLT2016(language_pair=('de', 'en'),
              valid_set='tst2013',
              test_set='tst2014',
              split=('train', 'valid', 'test'),
              root='.data',
              vocab=(None, None),
              tokenizer=None):
    """ Define translation datasets: IWSLT2016
    Separately returns train/valid/test datasets
    The available datasets include following:
    - language pairs
    [('en', 'ar'), ('en', 'de'), ('en', 'fr'), ('en', 'cs'), ('ar', 'en'),
    ('fr', 'en'), ('de', 'en'), ('cs', 'en')]
    - valid/test sets
    ['dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013', 'tst2014']
    For additional details refer to source website below:
    https://wit3.fbk.eu/2016-01

    Args:
        language_pair: tuple or list of two elements: src and tgt language
        valid_set: a string to identify validation set. The actual filenames would be
            'IWSLT16.TED.{}.{}-{}.{}'.format(valid_set,language_pair[0],language_pair[1],language_pair[0])
            and 'IWSLT16.TED.{}.{}-{}.{}'.format(valid_set,language_pair[0],language_pair[1],language_pair[1])
        test_set: a string to identify test set. The actual filenames would be as defined for valid_set
        split: a string or tuple for the returned datasets, Default: ('train', 'valid', 'test')
            By default, all the three datasets (train, valid, test) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test data.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
            get_tokenizer("spacy", language='en_core_web_sm'))


    Examples:
        >>> from torchtext.experimental.datasets import IWSLT
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = IWSLT(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """
    year = 16

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple")

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements'

    src_language, tgt_language = language_pair[0], language_pair[1]

    train_filenames = 'train.{}-{}.{}'.format(src_language, tgt_language, src_language), 'train.{}-{}.{}'.format(src_language, tgt_language, tgt_language)
    valid_filenames = 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, valid_set, src_language, tgt_language, src_language), 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, valid_set, src_language, tgt_language, tgt_language)
    test_filenames = 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, test_set, src_language, tgt_language, src_language), 'IWSLT{}.TED.{}.{}-{}.{}'.format(year, test_set, src_language, tgt_language, tgt_language)

    return _setup_datasets("IWSLT2016", train_filenames, valid_filenames, test_filenames,
                           split, root, vocab, tokenizer)


def WMT14(train_filenames=('train.tok.clean.bpe.32000.de',
                           'train.tok.clean.bpe.32000.en'),
          valid_filenames=('newstest2013.tok.bpe.32000.de',
                           'newstest2013.tok.bpe.32000.en'),
          test_filenames=('newstest2014.tok.bpe.32000.de',
                          'newstest2014.tok.bpe.32000.en'),
          split=('train', 'valid', 'test'),
          root='.data',
          vocab=(None, None),
          tokenizer=None):
    """ Define translation datasets: WMT14
    Separately returns train/valid/test datasets
    The available datasets include:

            newstest2016.en
            newstest2016.de
            newstest2015.en
            newstest2015.de
            newstest2014.en
            newstest2014.de
            newstest2013.en
            newstest2013.de
            newstest2012.en
            newstest2012.de
            newstest2011.tok.de
            newstest2011.en
            newstest2011.de
            newstest2010.tok.de
            newstest2010.en
            newstest2010.de
            newstest2009.tok.de
            newstest2009.en
            newstest2009.de
            newstest2016.tok.de
            newstest2015.tok.de
            newstest2014.tok.de
            newstest2013.tok.de
            newstest2012.tok.de
            newstest2010.tok.en
            newstest2009.tok.en
            newstest2015.tok.en
            newstest2014.tok.en
            newstest2013.tok.en
            newstest2012.tok.en
            newstest2011.tok.en
            newstest2016.tok.en
            newstest2009.tok.bpe.32000.en
            newstest2011.tok.bpe.32000.en
            newstest2010.tok.bpe.32000.en
            newstest2013.tok.bpe.32000.en
            newstest2012.tok.bpe.32000.en
            newstest2015.tok.bpe.32000.en
            newstest2014.tok.bpe.32000.en
            newstest2016.tok.bpe.32000.en
            train.tok.clean.bpe.32000.en
            newstest2009.tok.bpe.32000.de
            newstest2010.tok.bpe.32000.de
            newstest2011.tok.bpe.32000.de
            newstest2013.tok.bpe.32000.de
            newstest2012.tok.bpe.32000.de
            newstest2014.tok.bpe.32000.de
            newstest2016.tok.bpe.32000.de
            newstest2015.tok.bpe.32000.de
            train.tok.clean.bpe.32000.de

    Args:
        train_filenames: the source and target filenames for training.
            Default: ('train.tok.clean.bpe.32000.de', 'train.tok.clean.bpe.32000.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('newstest2013.tok.bpe.32000.de', 'newstest2013.tok.bpe.32000.en')
        test_filenames: the source and target filenames for test.
            Default: ('newstest2014.tok.bpe.32000.de', 'newstest2014.tok.bpe.32000.en')
        split: a string or tuple for the returned datasets, Default: ('train', 'valid', 'test')
            By default, all the three datasets (train, valid, test) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test data.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
            get_tokenizer("spacy", language='en_core_web_sm'))

    Examples:
        >>> from torchtext.experimental.datasets import WMT14
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = WMT14(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """

    return _setup_datasets("WMT14", train_filenames, valid_filenames, test_filenames,
                           split, root, vocab, tokenizer)


DATASETS = {'Multi30k': Multi30k, 'IWSLT2016': IWSLT2016, 'IWSLT2017': IWSLT2017, 'WMT14': WMT14}
