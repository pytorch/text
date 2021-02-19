import os
import io
import codecs
import xml.etree.ElementTree as ET
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header

URL = 'https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8'

_PATH = '2016-01.tgz'

MD5 = 'c393ed3fc2a1b0f004b3331043f615ae'

NUM_LINES = {
    'train': 196884,
    'valid': 993,
    'test': 1305,
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
        for line in fd_orig:
            if not any(tag in line for tag in xml_tags):
                # TODO: Fix utf-8 next line mark
                #                fd_txt.write(l.strip() + '\n')
                #                fd_txt.write(l.strip() + u"\u0085")
                #                fd_txt.write(l.lstrip())
                fd_txt.write(line.strip() + '\n')


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


@wrap_split_argument
@add_docstring_header()
def IWSLT(root='.data', split=('train', 'valid', 'test'), offset=0,
          train_filenames=('train.de-en.de', 'train.de-en.en'),
          valid_filenames=('IWSLT16.TED.tst2013.de-en.de',
                           'IWSLT16.TED.tst2013.de-en.en'),
          test_filenames=('IWSLT16.TED.tst2014.de-en.de',
                          'IWSLT16.TED.tst2014.de-en.en')):
    """    train_filenames: the source and target filenames for training.
                Default: ('train.de-en.de', 'train.de-en.en')
            valid_filenames: the source and target filenames for valid.
                Default: ('IWSLT16.TED.tst2013.de-en.de', 'IWSLT16.TED.tst2013.de-en.en')
            test_filenames: the source and target filenames for test.
                Default: ('IWSLT16.TED.tst2014.de-en.de', 'IWSLT16.TED.tst2014.de-en.en')

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
    """
    if not isinstance(train_filenames, tuple) and not isinstance(valid_filenames, tuple) \
            and not isinstance(test_filenames, tuple):
        raise ValueError("All filenames must be tuples")
    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, path=os.path.join(root, _PATH), hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    extracted_files = []  # list of paths to the extracted files
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5,
                                    path=os.path.join(root, _PATH), hash_type='md5')
    extracted_dataset_tar = extract_archive(dataset_tar)
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
            RawTextIterableDataset("IWSLT", NUM_LINES[key], _iter(src_data_iter, tgt_data_iter), offset=offset))

    return datasets
