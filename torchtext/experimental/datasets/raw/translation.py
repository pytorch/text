import os
import io
import codecs
import xml.etree.ElementTree as ET
from collections import defaultdict
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header

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

_PATHS = {
    'WMT14': 'wmt16_en_de.tar.gz',
    'IWSLT': '2016-01.tgz'
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
                    split, root, offset):
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
        dataset_tar = download_from_url(URLS[dataset_name], root=root, hash_value=MD5[dataset_name], path=_PATHS[dataset_name], hash_type='md5')
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

    return datasets


@wrap_split_argument
@add_docstring_header()
def Multi30k(root='.data', split=('train', 'valid', 'test'), offset=0,
             train_filenames=("train.de", "train.en"),
             valid_filenames=("val.de", "val.en"),
             test_filenames=("test_2016_flickr.de", "test_2016_flickr.en")):
    """    train_filenames: the source and target filenames for training.
                Default: ('train.de', 'train.en')
            valid_filenames: the source and target filenames for valid.
                Default: ('val.de', 'val.en')
            test_filenames: the source and target filenames for test.
                Default: ('test2016.de', 'test2016.en')

    Examples:
        >>> from torchtext.experimental.datasets.raw import Multi30k
        >>> train_dataset, valid_dataset, test_dataset = Multi30k()
    """
    return _setup_datasets("Multi30k", train_filenames, valid_filenames, test_filenames, split, root, offset)


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

    Examples:
        >>> from torchtext.experimental.datasets.raw import IWSLT
        >>> train_dataset, valid_dataset, test_dataset = IWSLT()
    """
    return _setup_datasets("IWSLT", train_filenames, valid_filenames, test_filenames, split, root, offset)


@wrap_split_argument
@add_docstring_header()
def WMT14(root='.data', split=('train', 'valid', 'test'), offset=0,
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
