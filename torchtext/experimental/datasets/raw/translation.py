import torch
import os
import io
import codecs
import xml.etree.ElementTree as ET
from collections import defaultdict

from torchtext.utils import (download_from_url, extract_archive,
                             unicode_csv_reader)
from torchtext.experimental.raw.common import RawTextIterableDataset

URLS = {
    'Multi30k': [
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.cs.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.fr.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_flickr.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_flickr.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_flickr.fr.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_mscoco.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_mscoco.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_mscoco.fr.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.cs.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.fr.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.cs.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.fr.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.1.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.1.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.2.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.2.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.3.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.3.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.4.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.4.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.5.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.5.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.1.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.1.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.2.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.2.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.3.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.3.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.4.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.4.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.5.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.5.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.1.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.1.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.2.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.2.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.3.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.3.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.4.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.4.en.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.5.de.gz",
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.5.en.gz"
    ],
    'WMT14':
    'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8',
    'IWSLT':
    'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
}


def _read_text_iterator(path):
    with io.open(path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            yield " ".join(row)


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
                    train_filenames,
                    valid_filenames,
                    test_filenames,
                    root='.data'):
    if not isinstance(train_filenames, tuple) and not isinstance(valid_filenames, tuple) \
            and not isinstance(test_filenames, tuple):
        raise ValueError("All filenames must be tuples")

    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

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

    datasets = []
    for key in data_filenames.keys():
        src_data_iter = _read_text_iterator(data_filenames[key][0])
        tgt_data_iter = _read_text_iterator(data_filenames[key][1])

        datasets.append(
            RawTranslationIterableDataset(dataset_name, src_data_iter, tgt_data_iter))

    return tuple(datasets)


class RawTranslationIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, name, src_iterator, tgt_iterator):
        """Initiate text-classification dataset.
        """
        super(RawTranslationIterableDataset, self).__init__()
        self.name = name
        self._src_iterator = src_iterator
        self._tgt_iterator = tgt_iterator
        self.has_setup = False
        self.start = 0
        self.num_lines = None

    def setup_iter(self, start=0, num_lines=None):
        self.start = start
        self.num_lines = num_lines
        self.has_setup = True

    def __iter__(self):
        if not self.has_setup:
            self.setup_iter()

        for i, item in enumerate(zip(self._src_iterator, self._tgt_iterator)):
            if i >= self.start:
                yield item
            if (self.num_lines is not None) and (i == (self.start +
                                                       self.num_lines)):
                break

    def __len__(self):
        return NUM_LINES[self.name]

    def get_iterator(self):
        return (self._src_iterator, self._tgt_iterator)


def Multi30k(train_filenames=("train.de", "train.en"),
             valid_filenames=("val.de", "val.en"),
             test_filenames=("test_2016_flickr.de", "test_2016_flickr.en"),
             root='.data'):
    """ Define translation datasets: Multi30k
        Separately returns train/valid/test datasets as a tuple
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

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.de', 'train.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('val.de', 'val.en')
        test_filenames: the source and target filenames for test.
            Default: ('test2016.de', 'test2016.en')
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets import Multi30k
        >>> train_dataset, valid_dataset, test_dataset = Multi30k()
    """
    return _setup_datasets("Multi30k",
                           train_filenames=train_filenames,
                           valid_filenames=valid_filenames,
                           test_filenames=test_filenames,
                           root=root)


def IWSLT(train_filenames=('train.de-en.de', 'train.de-en.en'),
          valid_filenames=('IWSLT16.TED.tst2013.de-en.de',
                           'IWSLT16.TED.tst2013.de-en.en'),
          test_filenames=('IWSLT16.TED.tst2014.de-en.de',
                          'IWSLT16.TED.tst2014.de-en.en'),
          root='.data'):
    """ Define translation datasets: IWSLT
        Separately returns train/valid/test datasets
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

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.de-en.de', 'train.de-en.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('IWSLT16.TED.tst2013.de-en.de', 'IWSLT16.TED.tst2013.de-en.en')
        test_filenames: the source and target filenames for test.
            Default: ('IWSLT16.TED.tst2014.de-en.de', 'IWSLT16.TED.tst2014.de-en.en')
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets.raw import IWSLT
        >>> train_dataset, valid_dataset, test_dataset = IWSLT()
    """
    src_language = train_filenames[0].split(".")[-1]
    tgt_language = train_filenames[1].split(".")[-1]
    languages = "-".join([src_language, tgt_language])
    URLS["IWSLT"] = URLS["IWSLT"].format(src_language, tgt_language, languages)

    return _setup_datasets(
        "IWSLT",
        train_filenames=train_filenames,
        valid_filenames=valid_filenames,
        test_filenames=test_filenames,
        root=root,
    )


def WMT14(train_filenames=('train.tok.clean.bpe.32000.de',
                           'train.tok.clean.bpe.32000.en'),
          valid_filenames=('newstest2013.tok.bpe.32000.de',
                           'newstest2013.tok.bpe.32000.en'),
          test_filenames=('newstest2014.tok.bpe.32000.de',
                          'newstest2014.tok.bpe.32000.en'),
          root='.data'):
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
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets import WMT14
        >>> train_dataset, valid_dataset, test_dataset = WMT14()
    """

    return _setup_datasets("WMT14",
                           train_filenames=train_filenames,
                           valid_filenames=valid_filenames,
                           test_filenames=test_filenames,
                           root=root)


DATASETS = {
    'Multi30k': Multi30k,
    'IWSLT': IWSLT,
    'WMT14': WMT14
}
NUM_LINES = {
    'Multi30k': 29000,
    'IWSLT': 173939,
    'WMT14': 4500966,
}
