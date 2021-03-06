import torch
import logging
from torchtext.data.datasets_utils import _check_default_set
from torchtext.data.datasets_utils import _wrap_datasets
from torchtext import datasets as raw
from torchtext.experimental.datasets import raw as experimental_raw
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
                    split_, root, vocab, tokenizer, **kwargs):
    split = _check_default_set(split_, ('train', 'valid', 'test'), dataset_name)
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

    if dataset_name == 'Multi30k' or dataset_name == 'WMT14':
        raw_datasets = experimental_raw.DATASETS[dataset_name](split=split, root=root, **kwargs)
    else:
        raw_datasets = raw.DATASETS[dataset_name](split=split, root=root, **kwargs)
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

    return _wrap_datasets(tuple(datasets), split_)


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


def Multi30k(task='task1',
             language_pair=('de', 'en'),
             train_set="train",
             valid_set="val",
             test_set="test_2016_flickr",
             split=('train', 'valid', 'test'),
             root='.data',
             vocab=(None, None),
             tokenizer=None):
    """ Define translation datasets: Multi30k
    Separately returns train/valid/test datasets as a tuple

    The available datasets include following:

    **Language pairs (task1)**:

    +-----+-----+-----+-----+-----+
    |     |'en' |'cs' |'de' |'fr' |
    +-----+-----+-----+-----+-----+
    |'en' |     |   x |  x  |  x  |
    +-----+-----+-----+-----+-----+
    |'cs' |  x  |     |  x  |  x  |
    +-----+-----+-----+-----+-----+
    |'de' |  x  |   x |     |  x  |
    +-----+-----+-----+-----+-----+
    |'fr' |  x  |   x |  x  |     |
    +-----+-----+-----+-----+-----+

    **Language pairs (task2)**:

    +-----+-----+-----+
    |     |'en' |'de' |
    +-----+-----+-----+
    |'en' |     |   x |
    +-----+-----+-----+
    |'de' |  x  |     |
    +-----+-----+-----+

    For additional details refer to source: https://github.com/multi30k/dataset


    Args:
        task: Indicate the task
        language_pair: tuple or list containing src and tgt language
        train_set: A string to identify train set.
        valid_set: A string to identify validation set.
        test_set: A string to identify test set.
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
        >>> from torchtext.experimental.datasets import Multi30k
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = (get_tokenizer("spacy", language='de'),
                         get_tokenizer("basic_english"))
        >>> train_dataset, valid_dataset, test_dataset = Multi30k(tokenizer=tokenizer)
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """
    return _setup_datasets("Multi30k", split, root, vocab, tokenizer,
                           task=task,
                           language_pair=language_pair,
                           train_set=train_set,
                           valid_set=valid_set,
                           test_set=test_set)


def IWSLT2017(language_pair=('de', 'en'),
              split=('train', 'valid', 'test'),
              root='.data',
              vocab=(None, None),
              tokenizer=None):
    """ Define translation datasets: IWSLT2017
    Separately returns train/valid/test datasets

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |'en' |'nl' |'de' |'it' |'ro' |
    +-----+-----+-----+-----+-----+-----+
    |'en' |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'nl' |  x  |     |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'de' |  x  |   x |     |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'it' |  x  |   x |  x  |     |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'ro' |  x  |   x |  x  |  x  |     |
    +-----+-----+-----+-----+-----+-----+

    For additional details refer to source website: https://wit3.fbk.eu/2017-01

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

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple")

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements'

    return _setup_datasets("IWSLT2017", split, root, vocab, tokenizer, language_pair=language_pair)


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

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |'en' |'fr' |'de' |'cs' |'ar' |
    +-----+-----+-----+-----+-----+-----+
    |'en' |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'fr' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |'de' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |'cs' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |'ar' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+

    **valid/test sets**: ['dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013', 'tst2014']

    For additional details refer to source website: https://wit3.fbk.eu/2016-01

    Args:
        language_pair: tuple or list of two elements: src and tgt language
        valid_set: a string to identify validation set.
        test_set: a string to identify test set.
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

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple")

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements'

    return _setup_datasets("IWSLT2016", split, root, vocab, tokenizer, language_pair=language_pair, valid_set=valid_set, test_set=test_set)


def WMT14(language_pair=('de', 'en'),
          train_set='train.tok.clean.bpe.32000',
          valid_set='newstest2013.tok.bpe.32000',
          test_set='newstest2014.tok.bpe.32000',
          split=('train', 'valid', 'test'),
          root='.data',
          vocab=(None, None),
          tokenizer=None):
    """ Define translation datasets: WMT14
    Separately returns train/valid/test datasets

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+
    |     |'en' |'de' |
    +-----+-----+-----+
    |'en' |     |   x |
    +-----+-----+-----+
    |'de' |  x  |     |
    +-----+-----+-----+


    Args:
        language_pair: tuple or list containing src and tgt language
        train_set: A string to identify train set.
        valid_set: A string to identify validation set.
        test_set: A string to identify test set.
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

    return _setup_datasets("WMT14", split, root, vocab, tokenizer,
                           language_pair=language_pair,
                           train_set=train_set,
                           valid_set=valid_set,
                           test_set=test_set)


DATASETS = {'Multi30k': Multi30k, 'IWSLT2016': IWSLT2016, 'IWSLT2017': IWSLT2017, 'WMT14': WMT14}
