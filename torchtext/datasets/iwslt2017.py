import os
import io
import codecs
import xml.etree.ElementTree as ET
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument

SUPPORTED_DATASETS = {

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

URL = SUPPORTED_DATASETS['URL']
MD5 = SUPPORTED_DATASETS['MD5']

NUM_LINES = {
    'train': {
        'train': {
            ('en', 'nl'): 237240,
            ('en', 'de'): 206112,
            ('en', 'it'): 231619,
            ('en', 'ro'): 220538,
            ('ro', 'de'): 201455,
            ('ro', 'en'): 220538,
            ('ro', 'nl'): 206920,
            ('ro', 'it'): 217551,
            ('de', 'ro'): 201455,
            ('de', 'en'): 206112,
            ('de', 'nl'): 213628,
            ('de', 'it'): 205465,
            ('it', 'en'): 231619,
            ('it', 'nl'): 233415,
            ('it', 'de'): 205465,
            ('it', 'ro'): 217551,
            ('nl', 'de'): 213628,
            ('nl', 'en'): 237240,
            ('nl', 'it'): 233415,
            ('nl', 'ro'): 206920
        }
    },
    'valid': {
        'dev2010': {
            ('en', 'nl'): 1003,
            ('en', 'de'): 888,
            ('en', 'it'): 929,
            ('en', 'ro'): 914,
            ('ro', 'de'): 912,
            ('ro', 'en'): 914,
            ('ro', 'nl'): 913,
            ('ro', 'it'): 914,
            ('de', 'ro'): 912,
            ('de', 'en'): 888,
            ('de', 'nl'): 1001,
            ('de', 'it'): 923,
            ('it', 'en'): 929,
            ('it', 'nl'): 1001,
            ('it', 'de'): 923,
            ('it', 'ro'): 914,
            ('nl', 'de'): 1001,
            ('nl', 'en'): 1003,
            ('nl', 'it'): 1001,
            ('nl', 'ro'): 913
        },
        'tst2010': {
            ('en', 'nl'): 1777,
            ('en', 'de'): 1568,
            ('en', 'it'): 1566,
            ('en', 'ro'): 1678,
            ('ro', 'de'): 1677,
            ('ro', 'en'): 1678,
            ('ro', 'nl'): 1680,
            ('ro', 'it'): 1643,
            ('de', 'ro'): 1677,
            ('de', 'en'): 1568,
            ('de', 'nl'): 1779,
            ('de', 'it'): 1567,
            ('it', 'en'): 1566,
            ('it', 'nl'): 1669,
            ('it', 'de'): 1567,
            ('it', 'ro'): 1643,
            ('nl', 'de'): 1779,
            ('nl', 'en'): 1777,
            ('nl', 'it'): 1669,
            ('nl', 'ro'): 1680
        }
    },
    'test': {
        'dev2010': {
            ('en', 'nl'): 1003,
            ('en', 'de'): 888,
            ('en', 'it'): 929,
            ('en', 'ro'): 914,
            ('ro', 'de'): 912,
            ('ro', 'en'): 914,
            ('ro', 'nl'): 913,
            ('ro', 'it'): 914,
            ('de', 'ro'): 912,
            ('de', 'en'): 888,
            ('de', 'nl'): 1001,
            ('de', 'it'): 923,
            ('it', 'en'): 929,
            ('it', 'nl'): 1001,
            ('it', 'de'): 923,
            ('it', 'ro'): 914,
            ('nl', 'de'): 1001,
            ('nl', 'en'): 1003,
            ('nl', 'it'): 1001,
            ('nl', 'ro'): 913
        },
        'tst2010': {
            ('en', 'nl'): 1777,
            ('en', 'de'): 1568,
            ('en', 'it'): 1566,
            ('en', 'ro'): 1678,
            ('ro', 'de'): 1677,
            ('ro', 'en'): 1678,
            ('ro', 'nl'): 1680,
            ('ro', 'it'): 1643,
            ('de', 'ro'): 1677,
            ('de', 'en'): 1568,
            ('de', 'nl'): 1779,
            ('de', 'it'): 1567,
            ('it', 'en'): 1566,
            ('it', 'nl'): 1669,
            ('it', 'de'): 1567,
            ('it', 'ro'): 1643,
            ('nl', 'de'): 1779,
            ('nl', 'en'): 1777,
            ('nl', 'it'): 1669,
            ('nl', 'ro'): 1680
        }
    }
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
        '<translator', '<title', '<speaker', '<doc', '</doc'
    ]
    f_txt = f_orig.replace('.tags', '')
    with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt,\
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


@wrap_split_argument(('train', 'valid', 'test'))
def IWSLT2017(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en')):
    """IWSLT2017 dataset

    The available datasets include following:

    **Language pairs**: [('en', 'nl'), ('en', 'de'), ('en', 'it'), ('en', 'ro'), ('ro', 'de'),
    ('ro', 'en'), ('ro', 'nl'), ('ro', 'it'), ('de', 'ro'), ('de', 'en'),
    ('de', 'nl'), ('de', 'it'), ('it', 'en'), ('it', 'nl'), ('it', 'de'),
    ('it', 'ro'), ('nl', 'de'), ('nl', 'en'), ('nl', 'it'), ('nl', 'ro')]

    For additional details refer to source website: https://wit3.fbk.eu/2017-01

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language

    """

    valid_set = 'dev2010'
    test_set = 'tst2010'

    num_lines_set_identifier = {
        'train': 'train',
        'valid': valid_set,
        'test': test_set
    }

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple but got {} instead".format(type(language_pair)))

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    src_language, tgt_language = language_pair[0], language_pair[1]

    if src_language not in SUPPORTED_DATASETS['language_pair'].keys():
        raise ValueError("src_language '{}' is not valid for ISWLT_year {}. Supported source languages are {}".
                         format(src_language, year, SUPPORTED_DATASETS['language_pair'].keys()))

    if tgt_language not in SUPPORTED_DATASETS['language_pair'][src_language]:
        raise ValueError("tgt_language '{}' is not valid for give src_language '{}'. Supported target language are {}".
                         format(tgt_language, src_language, SUPPORTED_DATASETS['language_pair'][src_language]))

    train_filenames = ('train.{}-{}.{}'.format(src_language, tgt_language, src_language),
                       'train.{}-{}.{}'.format(src_language, tgt_language, tgt_language))
    valid_filenames = ('IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS['year'], valid_set, src_language, tgt_language, src_language),
                       'IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS['year'], valid_set, src_language, tgt_language, tgt_language))
    test_filenames = ('IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS['year'], test_set, src_language, tgt_language, src_language),
                      'IWSLT{}.TED.{}.{}-{}.{}'.format(SUPPORTED_DATASETS['year'], test_set, src_language, tgt_language, tgt_language))

    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    extracted_files = []  # list of paths to the extracted files
    dataset_tar = download_from_url(SUPPORTED_DATASETS['URL'], root=root, hash_value=SUPPORTED_DATASETS['MD5'], path=os.path.join(root, SUPPORTED_DATASETS['_PATH']), hash_type='md5')
    extracted_dataset_tar = extract_archive(dataset_tar)
    # IWSLT dataset's url downloads a multilingual tgz.
    # We need to take an extra step to pick out the specific language pair from it.
    src_language = train_filenames[0].split(".")[-1]
    tgt_language = train_filenames[1].split(".")[-1]

    iwslt_tar = os.path.join(root, SUPPORTED_DATASETS['_PATH'].split(".")[0], 'texts/DeEnItNlRo/DeEnItNlRo', 'DeEnItNlRo-DeEnItNlRo.tgz')
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

    src_data_iter = _read_text_iterator(data_filenames[split][0])
    tgt_data_iter = _read_text_iterator(data_filenames[split][1])

    def _iter(src_data_iter, tgt_data_iter):
        for item in zip(src_data_iter, tgt_data_iter):
            yield item

    return RawTextIterableDataset("IWSLT2017", NUM_LINES[split][num_lines_set_identifier[split]][language_pair], _iter(src_data_iter, tgt_data_iter))
