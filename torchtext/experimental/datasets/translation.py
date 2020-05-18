import torch
import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import os
import io
import codecs
import xml.etree.ElementTree as ET
from collections import defaultdict

URLS = {
    'Multi30k': [
        'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
        'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
        'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/'
        'mmt_task1_test2016.tar.gz'
    ],
    'WMT14':
    'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8',
    'IWSLT':
    'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
}


def _read_text_iterator(path, tokenizer):
    r"""Read text from path and yield a list of tokens based on the tokenizer
    Arguments:
        path: the file path.
        tokenizer: the tokenizer used to tokenize string text.
    Examples:
        >>> from torchtext.data.functional import read_text_iterator
        >>> tokenizer = get_tokenizer("basic_english")
        >>> list((read_text_iterator('.data/ptb.train.txt', tokenizer)))
            [['Sentencepiece', 'encode', 'as', 'pieces'], ['example', 'to', 'try!']]
    """

    with io.open(path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = tokenizer(' '.join(row))
            yield tokens


def _create_data_from_iterator(vocab, iterator, removed_tokens=None):
    r"""Yield a list of ids from an token iterator with a vocab.
    Arguments:
        vocab: the vocabulary convert token into id.
        iterator: the iterator yield a list of tokens.
        removed_tokens: removed tokens from output dataset (Default: None)
    Examples:
        >>> from torchtext.data.functional import simple_space_split
        >>> from torchtext.data.functional import create_data_from_iterator
        >>> vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
        >>> list(create_data_from_iterator(vocab,
        >>>                                simple_space_split(["Sentencepiece as pieces",
        >>>                                                   "as pieces"]))
        >>> [[0, 2, 3], [2, 3]]
    """

    for tokens in iterator:
        if removed_tokens is None:
            tokens = [vocab[token] for token in tokens]
        else:
            token_ids = list(
                filter(lambda x: x not in removed_tokens,
                       [vocab[token] for token in tokens]))
            tokens = token_ids
        if len(tokens) == 0:
            logging.info('Row contains no tokens.')
        yield tokens


def _clean_xml_file(f_xml):
    f_txt = os.path.splitext(f_xml)[0]
    with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
        root = ET.parse(f_xml).getroot()[0]
        for doc in root.findall('doc'):
            for e in doc.findall('seg'):
                fd_txt.write(e.text.strip() + '\n')


def _clean_tags_file(f_orig):
    xml_tags = [
        '<url', '<keywords', '<talkid', '<description', '<reviewer',
        '<translator', '<title', '<speaker'
    ]
    f_txt = f_orig.replace('.tags', '')
    with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
            io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
        for l in fd_orig:
            if not any(tag in l for tag in xml_tags):
                # TODO: Fix utf-8 next line mark
                #                fd_txt.write(l.strip() + '\n')
                #                fd_txt.write(l.strip() + u"\u0085")
                #                fd_txt.write(l.lstrip())
                fd_txt.write(l.strip() + '\n')


def _construct_filenames(filename, languages):
    filenames = []
    for lang in languages:
        filenames.append(filename + "." + lang)
    return filenames


def _construct_filepaths(paths, src_filename, tgt_filename):
    src_path = None
    tgt_path = None
    for p in paths:
        src_path = p if src_filename in p else src_path
        tgt_path = p if tgt_filename in p else tgt_path
    return (src_path, tgt_path)


def _setup_datasets(dataset_name,
                    languages,
                    train_filename,
                    valid_filename,
                    test_filename,
                    data_select=('train', 'test', 'valid'),
                    tokenizer=(get_tokenizer("spacy", language='de_core_news_sm'),
                               get_tokenizer("spacy", language='en_core_web_sm')),
                    root='.data',
                    vocab=(None, None),
                    removed_tokens=['<unk>']):
    src_ext, tgt_ext = languages.split("-")
    src_vocab, tgt_vocab = vocab
    src_tokenizer, tgt_tokenizer = tokenizer
    src_train, tgt_train = _construct_filenames(train_filename,
                                                (src_ext, tgt_ext))
    src_eval, tgt_eval = _construct_filenames(valid_filename,
                                              (src_ext, tgt_ext))
    src_test, tgt_test = _construct_filenames(test_filename,
                                              (src_ext, tgt_ext))

    extracted_files = []
    if isinstance(URLS[dataset_name], list):
        for f in URLS[dataset_name]:
            dataset_tar = download_from_url(f, root=root)
            extracted_files.extend(extract_archive(dataset_tar))
    elif isinstance(URLS[dataset_name], str):
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files.extend(extract_archive(dataset_tar))
    else:
        raise ValueError(
            "URLS for {} has to be in a form or list or string".format(
                dataset_name))

    # Clean the xml and tag file in the archives
    file_archives = []
    for fname in extracted_files:
        if 'xml' in fname:
            _clean_xml_file(fname)
            file_archives.append(os.path.splitext(fname)[0])
        elif "tags" in fname:
            _clean_tags_file(fname)
            file_archives.append(fname.replace('.tags', ''))
        else:
            file_archives.append(fname)

    data_filenames = defaultdict(dict)
    data_filenames = {
        "train": _construct_filepaths(file_archives, src_train, tgt_train),
        "valid": _construct_filepaths(file_archives, src_eval, tgt_eval),
        "test": _construct_filepaths(file_archives, src_test, tgt_test)
    }

    for key in data_filenames.keys():
        if len(data_filenames[key]) == 0 or data_filenames[key] is None:
            raise FileNotFoundError(
                "Files are not found for data type {}".format(key))

    if src_vocab is None:
        logging.info('Building src Vocab based on train data')
        src_vocab = build_vocab_from_iterator(
            _read_text_iterator(data_filenames["train"][0], src_tokenizer))
    else:
        if not isinstance(src_vocab, Vocab):
            raise TypeError("Passed src vocabulary is not of type Vocab")
    logging.info('src Vocab has {} entries'.format(len(src_vocab)))

    if tgt_vocab is None:
        logging.info('Building tgt Vocab based on train data')
        tgt_vocab = build_vocab_from_iterator(
            _read_text_iterator(data_filenames["train"][1], tgt_tokenizer))
    else:
        if not isinstance(tgt_vocab, Vocab):
            raise TypeError("Passed tgt vocabulary is not of type Vocab")
    logging.info('tgt Vocab has {} entries'.format(len(tgt_vocab)))

    logging.info('Building datasets for {}'.format(data_select))
    datasets = []
    for key in data_filenames.keys():
        if key not in data_select:
            continue

        src_data_iter = _create_data_from_iterator(
            src_vocab, _read_text_iterator(data_filenames[key][0],
                                           src_tokenizer), removed_tokens)
        src_data = [torch.tensor(t).long() for t in src_data_iter]

        tgt_data_iter = _create_data_from_iterator(
            tgt_vocab, _read_text_iterator(data_filenames[key][1],
                                           tgt_tokenizer), removed_tokens)
        tgt_data = [torch.tensor(t).long() for t in tgt_data_iter]
        datasets.append(
            TranslationDataset(list(zip(src_data, tgt_data)),
                               (src_vocab, tgt_vocab)))

    return tuple(datasets)


class TranslationDataset(torch.utils.data.IterableDataset):
    """Defines a dataset for translation.
       Currently, we only support the following datasets:
             - Multi30k
             - WMT14
             - IWSLT
    """
    def __init__(self, data, vocab):
        """Initiate translation dataset.

        Arguments:
            data: a tuple of source and target tensors, which include token ids
                numericalizing the string tokens.
                [(src_tensor0, tgt_tensor0), (src_tensor1, tgt_tensor1)]
            vocab: source and target Vocabulary object used for dataset.
                (src_vocab, tgt_vocab)

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
        self._data = data
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_vocab(self):
        return self._vocab


def Multi30k(languages="de-en",
             train_filename="train",
             valid_filename="val",
             test_filename="test2016",
             tokenizer=(get_tokenizer("spacy", language='de_core_news_sm'),
                        get_tokenizer("spacy", language='en_core_web_sm')),
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
            Default: (torchtext.data.utils.get_tokenizer("spacy", language='de'),
                      torchtext.data.utils.get_tokenizer("spacy", language='en'))
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
          tokenizer=(get_tokenizer("spacy", language='de_core_news_sm'),
                     get_tokenizer("spacy", language='en_core_web_sm')),
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
            Default: (torchtext.data.utils.get_tokenizer("spacy", language='de'),
                      torchtext.data.utils.get_tokenizer("spacy", language='en'))
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
          tokenizer=(get_tokenizer("spacy", language='de_core_news_sm'),
                     get_tokenizer("spacy", language='en_core_web_sm')),
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
            Default: (torchtext.data.utils.get_tokenizer("spacy", language='de'),
                      torchtext.data.utils.get_tokenizer("spacy", language='en'))
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
