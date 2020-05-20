import torch
import logging

from torchtext.experimental.datasets import raw
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


def vocab_func(vocab):
    def _forward(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return _forward


def totensor(dtype):
    def _forward(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return _forward


def build_vocab(data, transforms, index):
    tok_list = []
    for line in data:
        tok_list.append(transforms(line[index]))
    return build_vocab_from_iterator(tok_list)


def sequential_transforms(*transforms):
    def _forward(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return _forward


def _setup_datasets(dataset_name,
                    languages,
                    train_filename,
                    valid_filename,
                    test_filename,
                    data_select=('train', 'test', 'valid'),
                    root='.data',
                    vocab=(None, None),
                    tokenizer=None,
                    removed_tokens=['<unk>']):
    src_ext, tgt_ext = languages.split("-")
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
    train, val, test = DATASETS[dataset_name](languages=languages,
                                              train_filename=train_filename,
                                              valid_filename=valid_filename,
                                              test_filename=test_filename,
                                              root=root)
    raw_data = {
        "train": [line for line in train],
        "valid": [line for line in val],
        "test": [line for line in test]
    }
    src_text_vocab_transform = sequential_transforms(src_tokenizer)
    tgt_text_vocab_transform = sequential_transforms(tgt_tokenizer)

    if src_vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building src Vocab based on train data')
        src_vocab = build_vocab(raw_data["train"],
                                src_text_vocab_transform,
                                index=0)
    else:
        if not isinstance(src_vocab, Vocab):
            raise TypeError("Passed src vocabulary is not of type Vocab")
    logging.info('src Vocab has {} entries'.format(len(src_vocab)))

    if tgt_vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building tgt Vocab based on train data')
        tgt_vocab = build_vocab(raw_data["train"],
                                tgt_text_vocab_transform,
                                index=1)
    else:
        if not isinstance(tgt_vocab, Vocab):
            raise TypeError("Passed tgt vocabulary is not of type Vocab")
    logging.info('tgt Vocab has {} entries'.format(len(tgt_vocab)))

    logging.info('Building datasets for {}'.format(data_select))
    datasets = []
    for key in data_select:
        src_text_transform = sequential_transforms(src_text_vocab_transform,
                                                   vocab_func(src_vocab),
                                                   totensor(dtype=torch.long))
        tgt_text_transform = sequential_transforms(tgt_text_vocab_transform,
                                                   vocab_func(tgt_vocab),
                                                   totensor(dtype=torch.long))
        datasets.append(
            TranslationDataset(raw_data[key], (src_vocab, tgt_vocab),
                               (src_text_transform, tgt_text_transform)))

    return tuple(datasets)


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
        source = self.transforms[0](self.data[i][0])
        target = self.transforms[1](self.data[i][1])
        return (source, target)

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


def Multi30k(languages="de-en",
             train_filename="train",
             valid_filename="val",
             test_filename="test2016",
             tokenizer=None,
             root='.data',
             vocab=(None, None),
             removed_tokens=['<unk>']):
    """ Define translation datasets: Multi30k
        Separately returns train/valid/test datasets as a tuple

    Arguments:
        languages: The source and target languages for the datasets.
            Will be used as a suffix for train_filename, valid_filename,
            and test_filename. The first split (before -) is the source
            and the second split is the target.
            Default: 'de-en' for source-target languages.
        train_filename: The source and target filenames for training
            without the extension since it's already handled by languages
            parameter.
            Default: 'train'
        valid_filename: The source and target filenames for valid
            without the extension since it's already handled by languages
            parameter.
            Default: 'val'
        test_filename: The source and target filenames for test
            without the extension since it's already handled by languages
            parameter.
            Default: 'test2016'
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            Default: None
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')

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
                           languages=languages,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           tokenizer=tokenizer,
                           root=root,
                           vocab=vocab,
                           removed_tokens=removed_tokens)


def IWSLT(languages='de-en',
          train_filename='train.de-en',
          valid_filename='IWSLT16.TED.tst2013.de-en',
          test_filename='IWSLT16.TED.tst2014.de-en',
          tokenizer=None,
          root='.data',
          vocab=(None, None),
          removed_tokens=['<unk>']):
    """ Define translation datasets: IWSLT
        Separately returns train/valid/test datasets
        The available datasets include:

    Arguments:
        languages: The source and target languages for the datasets.
            Will be used as a suffix for train_filename, valid_filename,
            and test_filename. The first split (before -) is the source
            and the second split is the target.
            Default: 'de-en' for source-target languages.
        train_filename: The source and target filenames for training
            without the extension since it's already handled by languages
            parameter.
            Default: 'train.de-en'
        valid_filename: The source and target filenames for valid
            without the extension since it's already handled by languages
            parameter.
            Default: 'IWSLT16.TED.tst2013.de-en'
        test_filename: The source and target filenames for test
            without the extension since it's already handled by languages
            parameter.
            Default: 'IWSLT16.TED.tst2014.de-en'
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            Default: None
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')

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
    src_language, tgt_language = languages.split('-')
    URLS["IWSLT"] = URLS["IWSLT"].format(src_language, tgt_language, languages)

    return _setup_datasets("IWSLT",
                           languages=languages,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           tokenizer=tokenizer,
                           root=root,
                           vocab=vocab,
                           removed_tokens=removed_tokens)


def WMT14(languages="de-en",
          train_filename='train.tok.clean.bpe.32000',
          valid_filename='newstest2013.tok.bpe.32000',
          test_filename='newstest2014.tok.bpe.32000',
          tokenizer=None,
          root='.data',
          vocab=(None, None),
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
        languages: The source and target languages for the datasets.
            Will be used as a suffix for train_filename, valid_filename,
            and test_filename. The first split (before -) is the source
            and the second split is the target.
            Default: 'de-en' for source-target languages.
        train_filename: The source and target filenames for training
            without the extension since it's already handled by languages
            parameter.
            Default: 'train.tok.clean.bpe.32000'
        valid_filename: The source and target filenames for valid
            without the extension since it's already handled by languages
            parameter.
            Default: 'newstest2013.tok.bpe.32000'
        test_filename: The source and target filenames for test
            without the extension since it's already handled by languages
            parameter.
            Default: 'newstest2014.tok.bpe.32000'
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            Default: None
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set.
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
                           languages=languages,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           tokenizer=tokenizer,
                           root=root,
                           vocab=vocab,
                           removed_tokens=removed_tokens)


DATASETS = {'Multi30k': raw.Multi30k, 'IWSLT': raw.IWSLT, 'WMT14': raw.WMT14}
