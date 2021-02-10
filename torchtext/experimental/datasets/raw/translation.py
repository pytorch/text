import os
import io
import codecs
import xml.etree.ElementTree as ET
from collections import defaultdict
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import check_default_set
from torchtext.experimental.datasets.raw.common import wrap_datasets

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
    'https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8'
}


def _read_text_iterator(path):
    with io.open(path, encoding="utf8") as f:
        for row in f:
            yield row


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
                    train_filenames, valid_filenames, test_filenames,
                    split_, root, offset):
    split = check_default_set(split_, ('train', 'valid', 'test'), dataset_name)
    if not isinstance(train_filenames, tuple) and not isinstance(valid_filenames, tuple) \
            and not isinstance(test_filenames, tuple):
        raise ValueError("All filenames must be tuples")
    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    extracted_files = []  # list of paths to the extracted files
    if isinstance(URLS[dataset_name], list):
        for idx, f in enumerate(URLS[dataset_name]):
            dataset_tar = download_from_url(
                f, root=root, hash_value=MD5[dataset_name][idx], hash_type='md5')
            extracted_files.extend(extract_archive(dataset_tar))
    elif isinstance(URLS[dataset_name], str):
        dataset_tar = download_from_url(URLS[dataset_name], root=root, hash_value=MD5[dataset_name], hash_type='md5')
        extracted_dataset_tar = extract_archive(dataset_tar)
        if dataset_name == 'IWSLT':
            # IWSLT dataset's url downloads a multilingual tgz.
            # We need to take an extra step to pick out the specific language pair from it.
            src_language = train_filenames[0].split(".")[-1]
            tgt_language = train_filenames[1].split(".")[-1]
            languages = "-".join([src_language, tgt_language])
            iwslt_tar = '.data/2016-01/texts/{}/{}/{}.tgz'
            iwslt_tar = iwslt_tar.format(
                src_language, tgt_language, languages)
            extracted_dataset_tar = extract_archive(iwslt_tar)
        extracted_files.extend(extracted_dataset_tar)
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
    for key in split:
        src_data_iter = _read_text_iterator(data_filenames[key][0])
        tgt_data_iter = _read_text_iterator(data_filenames[key][1])

        def _iter(src_data_iter, tgt_data_iter):
            for item in zip(src_data_iter, tgt_data_iter):
                yield item

        datasets.append(
            RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name][key], _iter(src_data_iter, tgt_data_iter), offset=offset))

    return wrap_datasets(tuple(datasets), split_)


def Multi30k(train_filenames=("train.de", "train.en"),
             valid_filenames=("val.de", "val.en"),
             test_filenames=("test_2016_flickr.de", "test_2016_flickr.en"),
             split=('train', 'valid', 'test'), root='.data', offset=0):
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
            just a string 'train'.
        root: Directory where the datasets are saved. Default: ".data"
        offset: the number of the starting line. Default: 0

    Examples:
        >>> from torchtext.experimental.datasets.raw import Multi30k
        >>> train_dataset, valid_dataset, test_dataset = Multi30k()
    """
    return _setup_datasets("Multi30k", train_filenames, valid_filenames, test_filenames, split, root, offset)


def IWSLT(train_filenames=('train.de-en.de', 'train.de-en.en'),
          valid_filenames=('IWSLT16.TED.tst2013.de-en.de',
                           'IWSLT16.TED.tst2013.de-en.en'),
          test_filenames=('IWSLT16.TED.tst2014.de-en.de',
                          'IWSLT16.TED.tst2014.de-en.en'),
          split=('train', 'valid', 'test'), root='.data', offset=0):
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

    Args:
        train_filenames: the source and target filenames for training.
            Default: ('train.de-en.de', 'train.de-en.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('IWSLT16.TED.tst2013.de-en.de', 'IWSLT16.TED.tst2013.de-en.en')
        test_filenames: the source and target filenames for test.
            Default: ('IWSLT16.TED.tst2014.de-en.de', 'IWSLT16.TED.tst2014.de-en.en')
        split: a string or tuple for the returned datasets, Default: ('train', 'valid', 'test')
            By default, all the three datasets (train, valid, test) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'.
        root: Directory where the datasets are saved. Default: ".data"
        offset: the number of the starting line. Default: 0

    Examples:
        >>> from torchtext.experimental.datasets.raw import IWSLT
        >>> train_dataset, valid_dataset, test_dataset = IWSLT()
    """
    return _setup_datasets("IWSLT", train_filenames, valid_filenames, test_filenames, split, root, offset)


def WMT14(train_filenames=('train.tok.clean.bpe.32000.de',
                           'train.tok.clean.bpe.32000.en'),
          valid_filenames=('newstest2013.tok.bpe.32000.de',
                           'newstest2013.tok.bpe.32000.en'),
          test_filenames=('newstest2014.tok.bpe.32000.de',
                          'newstest2014.tok.bpe.32000.en'),
          split=('train', 'valid', 'test'), root='.data', offset=0):
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
            just a string 'train'.
        root: Directory where the datasets are saved. Default: ".data"
        offset: the number of the starting line. Default: 0

    Examples:
        >>> from torchtext.experimental.datasets.raw import WMT14
        >>> train_dataset, valid_dataset, test_dataset = WMT14()
    """
    return _setup_datasets("WMT14", train_filenames, valid_filenames, test_filenames, split, root, offset)


DATASETS = {
    'Multi30k': Multi30k,
    'IWSLT': IWSLT,
    'WMT14': WMT14
}
NUM_LINES = {
    'Multi30k': {'train': 29000, 'valid': 1014, 'test': 1000},
    'IWSLT': {'train': 196884, 'valid': 993, 'test': 1305},
    'WMT14': {'train': 4500966, 'valid': 3000, 'test': 3003}
}
MD5 = {
    'Multi30k': ['3104872229daa1bef3b401d44dd2220b',
                 'efd67d314d98489b716b145475101932',
                 'ff2c0fcb4893a13bd73414306bc250ae',
                 '08dc7cd4a662f31718412de95ca9bfe3',
                 '6a8d5c87f6ae19e3d35681aa6fd16571',
                 '005396bac545d880abe6f00bbb7dbbb4',
                 'cb09af7d2b501f9112f2d6a59fa1360d',
                 'e8cd6ec2bc8a11fc846fa48a46e3d0bb',
                 'a7b684e0edbef1d4a23660c8e8e743fd',
                 '4995d10954a804d3cdfd907b9fd093e8',
                 'a152878809942757a55ce087073486b8',
                 'd9a5fc268917725a2b0efce3a0cc8607',
                 '81ff90b99829c0cd4b1b587d394afd39',
                 '0065d13af80720a55ca8153d126e6627',
                 '6cb767741dcad3931f966fefbc05203f',
                 '83cdc082f646b769095615384cf5c0ca',
                 '6e0e229eb049e3fc99a1ef02fb2d5f91',
                 '2b69aa9253948ac9f67e94917272dd40',
                 '93fc564584b7e5ba410c761ea5a1c682',
                 'ac0c72653c140dd96707212a1baa4278',
                 'eec05227daba4bb8f3f8f25b1cb335f4',
                 '6dfb42cae4e4fd9a3c40e62ff5398a55',
                 '9318fa08c0c0b96114eadb10eb2fc633',
                 'ece8cec6b87bf00dd12607f3062dae4c',
                 '088ec0765fa213a0eb937a62adfd4996',
                 '9a7e7b2dcc33135a32cd621c3b37d2d8',
                 '5f7c8d0be0ac739856b47d32a9434998',
                 '7d5ef0f069ee2d74dc2fdc6b46cd47fa',
                 '713ed720636622a54546d5f14f88b00f',
                 '62f36422bfab90fb42a560546b704009',
                 'cbf5bfc2147706f228d288e1b18bf4af',
                 '540da4566bb6dd35fdbc720218b742b7',
                 'bdfe4222f4692ccaa1e3389460f0890e',
                 '613eb4a3f0c2b13f0871ced946851b0e',
                 '0e1ee2b4145795bd180b193424db204b',
                 'd848fe0ae8b9447209fb49c5c31cb3d2',
                 '1cff688d1aadef7fdb22e9ad27d6fd2c',
                 'abc13b4042f4fef1cdff6de3b6c53b71',
                 '3e10289959d0059952511c31df3c7550',
                 'b26486ede1d4436d5acf6e38c65bb44d',
                 'df57faf5f00d434d2559c021ef55f1aa',
                 '16165248083beacebfe18866d5f4f0ae',
                 '9077a5127480cc799116384de501bd70',
                 '7180780822d4b600eb81c1ccf171c230',
                 'c1f697c3b6dfb7305349db34e26b45fc',
                 '8edb43c90cae66ec762748a968089b99',
                 'acb5ea26a577ceccfae6337181c31716',
                 '873a377a348713d3ab84db1fb57cdede',
                 '680816e0938fea5cf5331444bc09a4cf'],
    'IWSLT': 'c393ed3fc2a1b0f004b3331043f615ae',
    'WMT14': '874ab6bbfe9c21ec987ed1b9347f95ec'
}
