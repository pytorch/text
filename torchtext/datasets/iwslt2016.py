import os
import io
import codecs
import xml.etree.ElementTree as ET
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument


SUPPORTED_DATASETS = {
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

}

URL = SUPPORTED_DATASETS['URL']
MD5 = SUPPORTED_DATASETS['MD5']

NUM_LINES = {
    'train': {
        'train': {
            ('en', 'ar'): 224126,
            ('en', 'de'): 196884,
            ('en', 'fr'): 220400,
            ('en', 'cs'): 114390,
            ('ar', 'en'): 224126,
            ('fr', 'en'): 220400,
            ('de', 'en'): 196884,
            ('cs', 'en'): 114390
        }
    },
    'valid': {
        'dev2010': {
            ('en', 'ar'): 887,
            ('en', 'de'): 887,
            ('en', 'fr'): 887,
            ('en', 'cs'): 480,
            ('ar', 'en'): 887,
            ('fr', 'en'): 887,
            ('de', 'en'): 887,
            ('cs', 'en'): 480
        },
        'tst2010': {
            ('en', 'ar'): 1569,
            ('en', 'de'): 1565,
            ('en', 'fr'): 1664,
            ('en', 'cs'): 1511,
            ('ar', 'en'): 1569,
            ('fr', 'en'): 1664,
            ('de', 'en'): 1565,
            ('cs', 'en'): 1511
        },
        'tst2011': {
            ('en', 'ar'): 1199,
            ('en', 'de'): 1433,
            ('en', 'fr'): 818,
            ('en', 'cs'): 1013,
            ('ar', 'en'): 1199,
            ('fr', 'en'): 818,
            ('de', 'en'): 1433,
            ('cs', 'en'): 1013
        },
        'tst2012': {
            ('en', 'ar'): 1702,
            ('en', 'de'): 1700,
            ('en', 'fr'): 1124,
            ('en', 'cs'): 1385,
            ('ar', 'en'): 1702,
            ('fr', 'en'): 1124,
            ('de', 'en'): 1700,
            ('cs', 'en'): 1385
        },
        'tst2013': {
            ('en', 'ar'): 1169,
            ('en', 'de'): 993,
            ('en', 'fr'): 1026,
            ('en', 'cs'): 1327,
            ('ar', 'en'): 1169,
            ('fr', 'en'): 1026,
            ('de', 'en'): 993,
            ('cs', 'en'): 1327
        },
        'tst2014': {
            ('en', 'ar'): 1107,
            ('en', 'de'): 1305,
            ('en', 'fr'): 1305,
            ('ar', 'en'): 1107,
            ('fr', 'en'): 1305,
            ('de', 'en'): 1305
        }
    },
    'test': {
        'dev2010': {
            ('en', 'ar'): 887,
            ('en', 'de'): 887,
            ('en', 'fr'): 887,
            ('en', 'cs'): 480,
            ('ar', 'en'): 887,
            ('fr', 'en'): 887,
            ('de', 'en'): 887,
            ('cs', 'en'): 480
        },
        'tst2010': {
            ('en', 'ar'): 1569,
            ('en', 'de'): 1565,
            ('en', 'fr'): 1664,
            ('en', 'cs'): 1511,
            ('ar', 'en'): 1569,
            ('fr', 'en'): 1664,
            ('de', 'en'): 1565,
            ('cs', 'en'): 1511
        },
        'tst2011': {
            ('en', 'ar'): 1199,
            ('en', 'de'): 1433,
            ('en', 'fr'): 818,
            ('en', 'cs'): 1013,
            ('ar', 'en'): 1199,
            ('fr', 'en'): 818,
            ('de', 'en'): 1433,
            ('cs', 'en'): 1013
        },
        'tst2012': {
            ('en', 'ar'): 1702,
            ('en', 'de'): 1700,
            ('en', 'fr'): 1124,
            ('en', 'cs'): 1385,
            ('ar', 'en'): 1702,
            ('fr', 'en'): 1124,
            ('de', 'en'): 1700,
            ('cs', 'en'): 1385
        },
        'tst2013': {
            ('en', 'ar'): 1169,
            ('en', 'de'): 993,
            ('en', 'fr'): 1026,
            ('en', 'cs'): 1327,
            ('ar', 'en'): 1169,
            ('fr', 'en'): 1026,
            ('de', 'en'): 993,
            ('cs', 'en'): 1327
        },
        'tst2014': {
            ('en', 'ar'): 1107,
            ('en', 'de'): 1305,
            ('en', 'fr'): 1305,
            ('ar', 'en'): 1107,
            ('fr', 'en'): 1305,
            ('de', 'en'): 1305
        }
    }
}

SET_NOT_EXISTS = {
    ('en', 'ar'): [],
    ('en', 'de'): [],
    ('en', 'fr'): [],
    ('en', 'cs'): ['tst2014'],
    ('ar', 'en'): [],
    ('fr', 'en'): [],
    ('de', 'en'): [],
    ('cs', 'en'): ['tst2014']
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


# @add_docstring_header()
@wrap_split_argument(('train', 'valid', 'test'))
def IWSLT2016(root='.data', split=('train', 'valid', 'test'), offset=0, language_pair=('de', 'en'), valid_set='dev2010', test_set='tst2010'):
    """IWSLT2016 dataset

    The available datasets include following:

    **Language pairs**: [('en', 'ar'), ('en', 'de'), ('en', 'fr'), ('en', 'cs'), ('ar', 'en'),
    ('fr', 'en'), ('de', 'en'), ('cs', 'en')]

    **valid/test sets**: ['dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013', 'tst2014']

    For additional details refer to source website: https://wit3.fbk.eu/2016-01

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language
        valid_set: a string to identify validation set.
        test_set: a string to identify test set.

    """
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
        raise ValueError("src_language '{}' is not valid. Supported source languages are {}".
                         format(src_language, SUPPORTED_DATASETS['language_pair'].keys()))

    if tgt_language not in SUPPORTED_DATASETS['language_pair'][src_language]:
        raise ValueError("tgt_language '{}' is not valid for give src_language '{}'. Supported target language are {}".
                         format(tgt_language, src_language, SUPPORTED_DATASETS['language_pair'][src_language]))

    if valid_set not in SUPPORTED_DATASETS['valid_test'] or valid_set in SET_NOT_EXISTS[language_pair]:
        raise ValueError("valid_set '{}' is not valid for given language pair {}. Supported validation sets are {}".
                         format(valid_set, language_pair, [s for s in SUPPORTED_DATASETS['valid_test'] if s not in SET_NOT_EXISTS[language_pair]]))

    if test_set not in SUPPORTED_DATASETS['valid_test'] or test_set in SET_NOT_EXISTS[language_pair]:
        raise ValueError("test_set '{}' is not valid for give language pair {}. Supported test sets are {}".
                         format(valid_set, language_pair, [s for s in SUPPORTED_DATASETS['valid_test'] if s not in SET_NOT_EXISTS[language_pair]]))

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
    dataset_tar = download_from_url(SUPPORTED_DATASETS['URL'], root=root, hash_value=SUPPORTED_DATASETS['MD5'],
                                    path=os.path.join(root, SUPPORTED_DATASETS['_PATH']), hash_type='md5')
    extracted_dataset_tar = extract_archive(dataset_tar)
    # IWSLT dataset's url downloads a multilingual tgz.
    # We need to take an extra step to pick out the specific language pair from it.
    src_language = train_filenames[0].split(".")[-1]
    tgt_language = train_filenames[1].split(".")[-1]
    languages = "-".join([src_language, tgt_language])

    iwslt_tar = '{}/{}/texts/{}/{}/{}.tgz'
    iwslt_tar = iwslt_tar.format(
        root, SUPPORTED_DATASETS['_PATH'].split(".")[0], src_language, tgt_language, languages)
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

    return RawTextIterableDataset("IWSLT2016", NUM_LINES[split][num_lines_set_identifier[split]][language_pair], _iter(src_data_iter, tgt_data_iter))
