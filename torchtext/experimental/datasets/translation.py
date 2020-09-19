import torch
import logging

from torchtext.experimental.datasets import raw
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from ..functional import vocab_func, totensor, sequential_transforms


def build_vocab(data, transforms, index):
    tok_list = []
    for line in data:
        tok_list.append(transforms(line[index]))
    return build_vocab_from_iterator(tok_list)


def _setup_datasets(dataset_name,
                    train_filenames,
                    valid_filenames,
                    test_filenames,
                    data_select=('train', 'test', 'valid'),
                    root='.data',
                    vocab=(None, None),
                    tokenizer=(None, None),
                    removed_tokens=['<unk>']):
    if 'train' not in data_select and None in vocab:
        raise TypeError("If train is not selected must pass Vocab for both source and target.")
    if vocab[0] is not None and not isinstance(vocab[0], Vocab):
        raise TypeError("Passed src vocabulary is of type %s expected type Vocab".format(type(src_vocab)))
    if vocab[1] is not None and not isinstance(vocab[1], Vocab):
        raise TypeError("Passed tgt vocabulary is of type %s expected type Vocab".format(type(tgt_vocab)))
    if not isinstance(tokenizer, tuple) or len(tokenizer) != 2:
        raise ValueError("tokenizer must be tuple of length two. One for "
                         "source and target respectively. %s passed instead".format(tokenizer))
    tokenizer = (
        get_tokenizer("spacy", language='de_core_news_sm') if tokenizer[0] is None else tokenizer[0],
        get_tokenizer("spacy", language='en_core_web_sm') if tokenizer[1] is None else tokenizer[1])

    def build_raw_iter(raw_iter=None):
        raw_iter_ = raw.translation.DATASETS[dataset_name](train_filenames=train_filenames,
                                               valid_filenames=valid_filenames,
                                               test_filenames=test_filenames,
                                               root=root, data_select=data_select)
        if raw_iter is None:
            raw_iter = {}
        for i, name in enumerate(data_select):
            if name not in raw_iter:
                raw_iter[name] = raw_iter_[i]
        return raw_iter
    raw_iter = build_raw_iter()

    # pop train iterator to force repopulation
    vocab_ = len(vocab) * [None]
    for i in range(len(vocab)):
        vocab_[i] = build_vocab(raw_iter.pop("train"),
                               tokenizer[i],
                               index=i)
        raw_iter = build_raw_iter()
    vocab = tuple(vocab_)
    logging.info('src Vocab has {} entries'.format(len(vocab[0])))
    logging.info('tgt Vocab has {} entries'.format(len(vocab[1])))

    raw_data = {}
    for name in raw_iter:
        raw_data[name] = list(raw_iter[name])

    def build_transform(vocab, tokenizer):
        def fn(line):
            ids = []
            for token in tokenizer(line):
                ids.append(vocab[token])
            return torch.tensor(ids, dtype=torch.long)
        return fn

    logging.info('Building datasets for {}'.format(data_select))
    transforms = tuple(build_transform(v, t) for v, t in zip(vocab, tokenizer))
    return tuple(TranslationDataset(data, vocab, transforms)
                 for data in raw_data.values())


class TranslationDataset(torch.utils.data.Dataset):
    """Defines a dataset for translation.
       Currently, we only support the following datasets:
             - Multi30k
             - WMT14
             - IWSLT
    """

    def __init__(self, data, vocab, transforms):
        """Initiate translation dataset.

        Arguments:
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
        return tuple(map(self.transforms, self.data[i]))

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


def Multi30k(train_filenames=("train.de", "train.en"),
             valid_filenames=("val.de", "val.en"),
             test_filenames=("test_2016_flickr.de", "test_2016_flickr.en"),
             tokenizer=(None, None),
             root='.data',
             vocab=(None, None),
             data_select=('train', 'valid', 'test'),
             removed_tokens=['<unk>']):
    """ Define translation datasets: Multi30k
        Separately returns train/valid/test datasets as a tuple

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.de', 'train.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('val.de', 'val.en')
        test_filenames: the source and target filenames for test.
            Default: ('test2016.de', 'test2016.en')
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
                      get_tokenizer("spacy", language='en_core_web_sm'))
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'valid', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')
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
        >>> from torchtext.datasets import Multi30k
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = (get_tokenizer("spacy", language='de'),
                         get_tokenizer("basic_english"))
        >>> train_dataset, valid_dataset, test_dataset = Multi30k(tokenizer=tokenizer)
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """
    return _setup_datasets("Multi30k",
                           train_filenames=train_filenames,
                           valid_filenames=valid_filenames,
                           test_filenames=test_filenames,
                           tokenizer=tokenizer,
                           root=root,
                           data_select=data_select,
                           vocab=vocab,
                           removed_tokens=removed_tokens)


def IWSLT(train_filenames=('train.de-en.de', 'train.de-en.en'),
          valid_filenames=('IWSLT16.TED.tst2013.de-en.de',
                           'IWSLT16.TED.tst2013.de-en.en'),
          test_filenames=('IWSLT16.TED.tst2014.de-en.de',
                          'IWSLT16.TED.tst2014.de-en.en'),
          tokenizer=(None, None),
          root='.data',
          vocab=(None, None),
          data_select=('train', 'valid', 'test'),
          removed_tokens=['<unk>']):
    """ Define translation datasets: IWSLT
        Separately returns train/valid/test datasets
        The available datasets include:

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.de-en.de', 'train.de-en.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('IWSLT16.TED.tst2013.de-en.de', 'IWSLT16.TED.tst2013.de-en.en')
        test_filenames: the source and target filenames for test.
            Default: ('IWSLT16.TED.tst2014.de-en.de', 'IWSLT16.TED.tst2014.de-en.en')
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
                      get_tokenizer("spacy", language='en_core_web_sm'))
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'valid', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')
        The available datasets include:
            IWSLT16.TED.dev2010.ar-en.ar
            IWSLT16.TED.dev2010.ar-en.en
            IWSLT16.TED.dev2010.cs-en.cs
            IWSLT16.TED.dev2010.cs-en.en
            IWSLT16.TED.dev2010.de-en.de
            IWSLT16.TED.dev2010.de-en.en
            IWSLT16.TED.dev2010.en-ar.ar
            IWSLT16.TED.dev2010.en-ar.en
            IWSLT16.TED.dev2010.en-cs.cs
            IWSLT16.TED.dev2010.en-cs.en
            IWSLT16.TED.dev2010.en-de.de
            IWSLT16.TED.dev2010.en-de.en
            IWSLT16.TED.dev2010.en-fr.en
            IWSLT16.TED.dev2010.en-fr.fr
            IWSLT16.TED.dev2010.fr-en.en
            IWSLT16.TED.dev2010.fr-en.fr
            IWSLT16.TED.tst2010.ar-en.ar
            IWSLT16.TED.tst2010.ar-en.en
            IWSLT16.TED.tst2010.cs-en.cs
            IWSLT16.TED.tst2010.cs-en.en
            IWSLT16.TED.tst2010.de-en.de
            IWSLT16.TED.tst2010.de-en.en
            IWSLT16.TED.tst2010.en-ar.ar
            IWSLT16.TED.tst2010.en-ar.en
            IWSLT16.TED.tst2010.en-cs.cs
            IWSLT16.TED.tst2010.en-cs.en
            IWSLT16.TED.tst2010.en-de.de
            IWSLT16.TED.tst2010.en-de.en
            IWSLT16.TED.tst2010.en-fr.en
            IWSLT16.TED.tst2010.en-fr.fr
            IWSLT16.TED.tst2010.fr-en.en
            IWSLT16.TED.tst2010.fr-en.fr
            IWSLT16.TED.tst2011.ar-en.ar
            IWSLT16.TED.tst2011.ar-en.en
            IWSLT16.TED.tst2011.cs-en.cs
            IWSLT16.TED.tst2011.cs-en.en
            IWSLT16.TED.tst2011.de-en.de
            IWSLT16.TED.tst2011.de-en.en
            IWSLT16.TED.tst2011.en-ar.ar
            IWSLT16.TED.tst2011.en-ar.en
            IWSLT16.TED.tst2011.en-cs.cs
            IWSLT16.TED.tst2011.en-cs.en
            IWSLT16.TED.tst2011.en-de.de
            IWSLT16.TED.tst2011.en-de.en
            IWSLT16.TED.tst2011.en-fr.en
            IWSLT16.TED.tst2011.en-fr.fr
            IWSLT16.TED.tst2011.fr-en.en
            IWSLT16.TED.tst2011.fr-en.fr
            IWSLT16.TED.tst2012.ar-en.ar
            IWSLT16.TED.tst2012.ar-en.en
            IWSLT16.TED.tst2012.cs-en.cs
            IWSLT16.TED.tst2012.cs-en.en
            IWSLT16.TED.tst2012.de-en.de
            IWSLT16.TED.tst2012.de-en.en
            IWSLT16.TED.tst2012.en-ar.ar
            IWSLT16.TED.tst2012.en-ar.en
            IWSLT16.TED.tst2012.en-cs.cs
            IWSLT16.TED.tst2012.en-cs.en
            IWSLT16.TED.tst2012.en-de.de
            IWSLT16.TED.tst2012.en-de.en
            IWSLT16.TED.tst2012.en-fr.en
            IWSLT16.TED.tst2012.en-fr.fr
            IWSLT16.TED.tst2012.fr-en.en
            IWSLT16.TED.tst2012.fr-en.fr
            IWSLT16.TED.tst2013.ar-en.ar
            IWSLT16.TED.tst2013.ar-en.en
            IWSLT16.TED.tst2013.cs-en.cs
            IWSLT16.TED.tst2013.cs-en.en
            IWSLT16.TED.tst2013.de-en.de
            IWSLT16.TED.tst2013.de-en.en
            IWSLT16.TED.tst2013.en-ar.ar
            IWSLT16.TED.tst2013.en-ar.en
            IWSLT16.TED.tst2013.en-cs.cs
            IWSLT16.TED.tst2013.en-cs.en
            IWSLT16.TED.tst2013.en-de.de
            IWSLT16.TED.tst2013.en-de.en
            IWSLT16.TED.tst2013.en-fr.en
            IWSLT16.TED.tst2013.en-fr.fr
            IWSLT16.TED.tst2013.fr-en.en
            IWSLT16.TED.tst2013.fr-en.fr
            IWSLT16.TED.tst2014.ar-en.ar
            IWSLT16.TED.tst2014.ar-en.en
            IWSLT16.TED.tst2014.de-en.de
            IWSLT16.TED.tst2014.de-en.en
            IWSLT16.TED.tst2014.en-ar.ar
            IWSLT16.TED.tst2014.en-ar.en
            IWSLT16.TED.tst2014.en-de.de
            IWSLT16.TED.tst2014.en-de.en
            IWSLT16.TED.tst2014.en-fr.en
            IWSLT16.TED.tst2014.en-fr.fr
            IWSLT16.TED.tst2014.fr-en.en
            IWSLT16.TED.tst2014.fr-en.fr
            IWSLT16.TEDX.dev2012.de-en.de
            IWSLT16.TEDX.dev2012.de-en.en
            IWSLT16.TEDX.tst2013.de-en.de
            IWSLT16.TEDX.tst2013.de-en.en
            IWSLT16.TEDX.tst2014.de-en.de
            IWSLT16.TEDX.tst2014.de-en.en
            train.ar
            train.ar-en.ar
            train.ar-en.en
            train.cs
            train.cs-en.cs
            train.cs-en.en
            train.de
            train.de-en.de
            train.de-en.en
            train.en
            train.en-ar.ar
            train.en-ar.en
            train.en-cs.cs
            train.en-cs.en
            train.en-de.de
            train.en-de.en
            train.en-fr.en
            train.en-fr.fr
            train.fr
            train.fr-en.en
            train.fr-en.fr
            train.tags.ar-en.ar
            train.tags.ar-en.en
            train.tags.cs-en.cs
            train.tags.cs-en.en
            train.tags.de-en.de
            train.tags.de-en.en
            train.tags.en-ar.ar
            train.tags.en-ar.en
            train.tags.en-cs.cs
            train.tags.en-cs.en
            train.tags.en-de.de
            train.tags.en-de.en
            train.tags.en-fr.en
            train.tags.en-fr.fr
            train.tags.fr-en.en
            train.tags.fr-en.fr

    Examples:
        >>> from torchtext.datasets import IWSLT
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = IWSLT(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """

    return _setup_datasets("IWSLT",
                           train_filenames=train_filenames,
                           valid_filenames=valid_filenames,
                           test_filenames=test_filenames,
                           tokenizer=tokenizer,
                           root=root,
                           data_select=data_select,
                           vocab=vocab,
                           removed_tokens=removed_tokens)


def WMT14(train_filenames=('train.tok.clean.bpe.32000.de',
                           'train.tok.clean.bpe.32000.en'),
          valid_filenames=('newstest2013.tok.bpe.32000.de',
                           'newstest2013.tok.bpe.32000.en'),
          test_filenames=('newstest2014.tok.bpe.32000.de',
                          'newstest2014.tok.bpe.32000.en'),
          tokenizer=(None, None),
          root='.data',
          vocab=(None, None),
          data_select=('train', 'valid', 'test'),
          removed_tokens=['<unk>']):
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

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.tok.clean.bpe.32000.de', 'train.tok.clean.bpe.32000.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('newstest2013.tok.bpe.32000.de', 'newstest2013.tok.bpe.32000.en')
        test_filenames: the source and target filenames for test.
            Default: ('newstest2014.tok.bpe.32000.de', 'newstest2014.tok.bpe.32000.en')
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            It has to be in a form of tuple.
            Default: (get_tokenizer("spacy", language='de_core_news_sm'),
                      get_tokenizer("spacy", language='en_core_web_sm'))
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set. It has to be
            in a form of tuple.
            Default: (None, None)
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'valid', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')

    Examples:
        >>> from torchtext.datasets import WMT14
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = WMT14(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """

    return _setup_datasets("WMT14",
                           train_filenames=train_filenames,
                           valid_filenames=valid_filenames,
                           test_filenames=test_filenames,
                           data_select=data_select,
                           tokenizer=tokenizer,
                           root=root,
                           vocab=vocab,
                           removed_tokens=removed_tokens)


DATASETS = {'Multi30k': Multi30k, 'IWSLT': IWSLT, 'WMT14': WMT14}
