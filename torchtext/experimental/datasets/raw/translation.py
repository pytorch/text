import torch
import os
import io
import codecs
import xml.etree.ElementTree as ET
from collections import defaultdict

from torchtext.utils import (download_from_url, extract_archive,
                             unicode_csv_reader)

URLS = {
    'Multi30k': [
        "https://drive.google.com/uc?export=download&id=1I6OJBRr2UForrT4ZMe3yuDklb_9toGHs",
        "https://drive.google.com/uc?export=download&id=1oklC2pNNbPAWjMYO3x6ok6o3Wn_w4ebe"
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
                    languages,
                    train_filename,
                    valid_filename,
                    test_filename,
                    root='.data'):
    src_ext, tgt_ext = languages.split("-")
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

    datasets = []
    for key in data_filenames.keys():
        src_data_iter = _read_text_iterator(data_filenames[key][0])
        tgt_data_iter = _read_text_iterator(data_filenames[key][1])

        datasets.append(RawTextIterableDataset(src_data_iter, tgt_data_iter))

    return tuple(datasets)


class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """
    def __init__(self, src_iterator, tgt_iterator):
        """Initiate text-classification dataset.
        """
        super(RawTextIterableDataset, self).__init__()
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

    def get_iterator(self):
        return self._iterator


def Multi30k(languages="de-en",
             train_filename="train",
             valid_filename="val",
             test_filename="test_2016_flickr",
             root='.data'):
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
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets import Multi30k
        >>> train_dataset, valid_dataset, test_dataset = Multi30k()
    """
    return _setup_datasets("Multi30k",
                           languages=languages,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           root=root)


def IWSLT(languages='de-en',
          train_filename='train.de-en',
          valid_filename='IWSLT16.TED.tst2013.de-en',
          test_filename='IWSLT16.TED.tst2014.de-en',
          root='.data'):
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
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets.raw import IWSLT
        >>> train_dataset, valid_dataset, test_dataset = IWSLT()
    """
    src_language, tgt_language = languages.split('-')
    URLS["IWSLT"] = URLS["IWSLT"].format(src_language, tgt_language, languages)

    return _setup_datasets(
        "IWSLT",
        languages=languages,
        train_filename=train_filename,
        valid_filename=valid_filename,
        test_filename=test_filename,
        root=root,
    )


def WMT14(languages="de-en",
          train_filename='train.tok.clean.bpe.32000',
          valid_filename='newstest2013.tok.bpe.32000',
          test_filename='newstest2014.tok.bpe.32000',
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
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets import WMT14
        >>> train_dataset, valid_dataset, test_dataset = WMT14()
    """

    return _setup_datasets("WMT14",
                           languages=languages,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           root=root)


DATASETS = {'Multi30k': Multi30k, 'IWSLT': IWSLT, 'WMT14': WMT14}
