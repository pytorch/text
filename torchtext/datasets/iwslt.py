import os
import io
import codecs
import xml.etree.ElementTree as ET
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header


SUPPORTED_DATASETS = {
    2016: {
        'URL': 'https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8',
        '_PATH': '2016-01.tgz',
        'MD5': 'c393ed3fc2a1b0f004b3331043f615ae',
        'valid_test': ['dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013', 'tst2014'],
        'language_pair': {
            'en': ['ar', 'de', 'fr', 'cs'],
            'ar': ['en'],
            'fr': ['en'],
            'de': ['en'],
            'cs': ['en'],
        },
        'year': 16,
    },
    2017: {
        'URL': 'https://drive.google.com/u/0/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp',
        '_PATH': '2017-01-trnmted.tgz',
        'MD5': 'aca701032b1c4411afc4d9fa367796ba',
        'valid_test': ['dev2010', 'tst2010'],
        'language_pair': {
            'en': ['nl', 'de', 'it', 'ro'],
            'ro': ['de', 'en', 'nl', 'it'],
            'de': ['ro', 'en', 'nl', 'it'],
            'it': ['en', 'nl', 'de', 'ro'],
            'nl': ['de', 'en', 'it', 'ro'],
        },
        'year': 17,
    }
}

URL = [v['URL'] for v in SUPPORTED_DATASETS.values()]
MD5 = [v['MD5'] for v in SUPPORTED_DATASETS.values()]

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
def IWSLT(root='.data', split=('train', 'valid', 'test'), offset=0, language_pair=('de', 'en'), year=2016, valid_set='dev2010', test_set='tst2010'):
    """TODO
    """

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple")

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements'

    src_language, tgt_language = language_pair[0], language_pair[1]

    if year not in SUPPORTED_DATASETS.keys():
        raise ValueError("year {} is not valid. Supported years are {}".format(year, list(SUPPORTED_DATASETS.keys())))

    if src_language not in SUPPORTED_DATASETS[year]['language_pair'].keys():
        raise ValueError("src_language '{}' is not valid for ISWLT_year {}. Supported source languages are {}".format(src_language, year, SUPPORTED_DATASETS[year]['language_pair'].keys()))

    if tgt_language not in SUPPORTED_DATASETS[year]['language_pair'][src_language]:
        raise ValueError("tgt_language '{}' is not valid for give src_language '{}'. Supported target language are {}".format(tgt_language, src_language, SUPPORTED_DATASETS[year]['language_pair'][src_language]))

    if valid_set not in SUPPORTED_DATASETS[year]['valid_test']:
        raise ValueError("valid_set '{}' is not valid for give ISWLT_year {}. Supported validation sets are {}".format(valid_set, year, SUPPORTED_DATASETS[year]['valid_test']))

    if test_set not in SUPPORTED_DATASETS[year]['valid_test']:
        raise ValueError("test_set '{}' is not valid for give ISWLT_year {}. Supported test sets are {}".format(valid_set, year, SUPPORTED_DATASETS[year]['valid_test']))

    train_filenames = 'train.{}-{}.{}'.format(src_language, tgt_language, src_language), 'train.{}-{}.{}'.format(src_language, tgt_language, tgt_language)
    valid_filenames = 'IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS[year]['year'], valid_set, src_language, tgt_language, src_language), 'IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS[year]['year'], valid_set, src_language, tgt_language, tgt_language)
    test_filenames = 'IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS[year]['year'], test_set, src_language, tgt_language, src_language), 'IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS[year]['year'], test_set, src_language, tgt_language, tgt_language)

    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    extracted_files = []  # list of paths to the extracted files
    dataset_tar = download_from_url(SUPPORTED_DATASETS[year]['URL'], root=root, hash_value=SUPPORTED_DATASETS[year]['MD5'], path=os.path.join(root, SUPPORTED_DATASETS[year]['_PATH']), hash_type='md5')
    extracted_dataset_tar = extract_archive(dataset_tar)
    # IWSLT dataset's url downloads a multilingual tgz.
    # We need to take an extra step to pick out the specific language pair from it.
    src_language = train_filenames[0].split(".")[-1]
    tgt_language = train_filenames[1].split(".")[-1]
    languages = "-".join([src_language, tgt_language])
    # For Year 2017 everything is put in the same folder
    if year == 2017:
        iwslt_tar = os.path.join(root, SUPPORTED_DATASETS[year]['_PATH'].split(".")[0], 'texts/DeEnItNlRo/DeEnItNlRo', 'DeEnItNlRo-DeEnItNlRo.tgz')
    else:
        iwslt_tar = '{}/{}/texts/{}/{}/{}.tgz'
        iwslt_tar = iwslt_tar.format(
            root, SUPPORTED_DATASETS[year]['_PATH'].split(".")[0], src_language, tgt_language, languages)
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
        assert (data_filenames[key][0] is not None), 'Internal Error: {} file not found'.format(key)
        assert (data_filenames[key][1] is not None), 'Internal Error: {} file not found'.format(key)
        src_data_iter = _read_text_iterator(data_filenames[key][0])
        tgt_data_iter = _read_text_iterator(data_filenames[key][1])

        def _iter(src_data_iter, tgt_data_iter):
            for item in zip(src_data_iter, tgt_data_iter):
                yield item

        datasets.append(
            RawTextIterableDataset("IWSLT", NUM_LINES[key], _iter(src_data_iter, tgt_data_iter), offset=offset))

    return datasets
