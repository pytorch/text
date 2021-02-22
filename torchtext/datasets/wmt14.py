import os
import io
import codecs
import xml.etree.ElementTree as ET
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header

URL = 'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'

_PATH = 'wmt16_en_de.tar.gz'

MD5 = '874ab6bbfe9c21ec987ed1b9347f95ec'

NUM_LINES = {
    'train': 4500966,
    'valid': 3000,
    'test': 3003,
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


@add_docstring_header(num_lines=NUM_LINES)
@wrap_split_argument(('train', 'valid', 'test'))
def WMT14(root, split,
          train_filenames=('train.tok.clean.bpe.32000.de',
                           'train.tok.clean.bpe.32000.en'),
          valid_filenames=('newstest2013.tok.bpe.32000.de',
                           'newstest2013.tok.bpe.32000.en'),
          test_filenames=('newstest2014.tok.bpe.32000.de',
                          'newstest2014.tok.bpe.32000.en')):
    """    train_filenames: the source and target filenames for training.
                Default: ('train.tok.clean.bpe.32000.de', 'train.tok.clean.bpe.32000.en')
            valid_filenames: the source and target filenames for valid.
                Default: ('newstest2013.tok.bpe.32000.de', 'newstest2013.tok.bpe.32000.en')
            test_filenames: the source and target filenames for test.
                Default: ('newstest2014.tok.bpe.32000.de', 'newstest2014.tok.bpe.32000.en')

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
    """
    if not isinstance(train_filenames, tuple) and not isinstance(valid_filenames, tuple) \
            and not isinstance(test_filenames, tuple):
        raise ValueError("All filenames must be tuples")
    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, path=os.path.join(root, _PATH), hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

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

    src_data_iter = _read_text_iterator(data_filenames[split][0])
    tgt_data_iter = _read_text_iterator(data_filenames[split][1])

    def _iter(src_data_iter, tgt_data_iter):
        for item in zip(src_data_iter, tgt_data_iter):
            yield item

    return RawTextIterableDataset("WMT14", NUM_LINES[split], _iter(src_data_iter, tgt_data_iter))
