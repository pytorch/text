from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper, FileLister

import os
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _clean_xml_file,
    _clean_tags_file,
)
from torchtext.data.datasets_utils import _create_dataset_directory

URL = 'https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8'

_PATH = '2016-01.tgz'

MD5 = 'c393ed3fc2a1b0f004b3331043f615ae'

SUPPORTED_DATASETS = {

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

NUM_LINES = {
    'train': {
        'train': {
            ('ar', 'en'): 224126,
            ('de', 'en'): 196884,
            ('en', 'fr'): 220400,
            ('cs', 'en'): 114390
        }
    },
    'valid': {
        'dev2010': {
            ('ar', 'en'): 887,
            ('de', 'en'): 887,
            ('en', 'fr'): 887,
            ('cs', 'en'): 480
        },
        'tst2010': {
            ('ar', 'en'): 1569,
            ('de', 'en'): 1565,
            ('en', 'fr'): 1664,
            ('cs', 'en'): 1511
        },
        'tst2011': {
            ('ar', 'en'): 1199,
            ('de', 'en'): 1433,
            ('en', 'fr'): 818,
            ('cs', 'en'): 1013
        },
        'tst2012': {
            ('ar', 'en'): 1702,
            ('de', 'en'): 1700,
            ('en', 'fr'): 1124,
            ('cs', 'en'): 1385
        },
        'tst2013': {
            ('ar', 'en'): 1169,
            ('de', 'en'): 993,
            ('en', 'fr'): 1026,
            ('cs', 'en'): 1327
        },
        'tst2014': {
            ('ar', 'en'): 1107,
            ('de', 'en'): 1305,
            ('en', 'fr'): 1305
        }
    },
    'test': {
        'dev2010': {
            ('ar', 'en'): 887,
            ('de', 'en'): 887,
            ('en', 'fr'): 887,
            ('cs', 'en'): 480
        },
        'tst2010': {
            ('ar', 'en'): 1569,
            ('de', 'en'): 1565,
            ('en', 'fr'): 1664,
            ('cs', 'en'): 1511
        },
        'tst2011': {
            ('ar', 'en'): 1199,
            ('de', 'en'): 1433,
            ('en', 'fr'): 818,
            ('cs', 'en'): 1013
        },
        'tst2012': {
            ('ar', 'en'): 1702,
            ('de', 'en'): 1700,
            ('en', 'fr'): 1124,
            ('cs', 'en'): 1385
        },
        'tst2013': {
            ('ar', 'en'): 1169,
            ('de', 'en'): 993,
            ('en', 'fr'): 1026,
            ('cs', 'en'): 1327
        },
        'tst2014': {
            ('ar', 'en'): 1107,
            ('de', 'en'): 1305,
            ('en', 'fr'): 1305
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


def _construct_filenames(filename, languages):
    filenames = []
    for lang in languages:
        filenames.append(filename + "." + lang)
    return filenames


def _construct_filepath(path, src_filename, tgt_filename):
    src_path = None
    tgt_path = None
    src_path = path if src_filename in path else src_path
    tgt_path = path if tgt_filename in path else tgt_path
    return src_path, tgt_path


def _construct_filepaths(paths, src_filename, tgt_filename):
    src_path = None
    tgt_path = None
    for p in paths:
        src_path, tgt_path = _construct_filepath(p, src_filename, tgt_filename)
    return src_path, tgt_path


DATASET_NAME = "IWSLT2016"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def IWSLT2016(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'), valid_set='tst2013', test_set='tst2014'):
    """IWSLT2016 dataset

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |'en' |'fr' |'de' |'cs' |'ar' |
    +-----+-----+-----+-----+-----+-----+
    |'en' |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'fr' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |'de' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |'cs' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |'ar' |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+

    **valid/test sets**: ['dev2010', 'tst2010', 'tst2011', 'tst2012', 'tst2013', 'tst2014']

    For additional details refer to source website: https://wit3.fbk.eu/2016-01

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language
        valid_set: a string to identify validation set.
        test_set: a string to identify test set.

    Examples:
        >>> from torchtext.datasets import IWSLT2016
        >>> train_iter, valid_iter, test_iter = IWSLT2016()
        >>> src_sentence, tgt_sentence = next(train_iter)

    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple but got {} instead".format(type(language_pair)))

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    src_language, tgt_language = language_pair[0], language_pair[1]

    if src_language not in SUPPORTED_DATASETS['language_pair']:
        raise ValueError("src_language '{}' is not valid. Supported source languages are {}".
                         format(src_language, list(SUPPORTED_DATASETS['language_pair'])))

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

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH),
        hash_dict={os.path.join(root, _PATH): MD5},
        hash_type="md5"
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_compressed_dp = FileOpener(cache_compressed_dp, mode="b")
    src_language = train_filenames[0].split(".")[-1]
    tgt_language = train_filenames[1].split(".")[-1]
    languages = "-".join([src_language, tgt_language])

    iwslt_tar = os.path.join(
        "texts", src_language, tgt_language, languages
    )
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(os.path.splitext(x[0])[0], iwslt_tar)
    )
    cache_decompressed_dp = cache_decompressed_dp.read_from_tar()
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb")

    def clean_files(fname):
        if 'xml' in fname:
            _clean_xml_file(fname)
            return os.path.splitext(fname)[0]
        elif "tags" in fname:
            _clean_tags_file(fname)
            return fname.replace('.tags', '')
        return fname

    cache_decompressed_dp = cache_decompressed_dp.flatmap(FileLister)

    def get_filepath(f):
        src_file, tgt_file = {
            "train": _construct_filepath(f, src_train, tgt_train),
            "valid": _construct_filepath(f, src_eval, tgt_eval),
            "test": _construct_filepath(f, src_test, tgt_test)
        }[split]

        return src_file, tgt_file

    cleaned_cache_decompressed_dp = cache_decompressed_dp.map(clean_files).map(get_filepath)

    # pairs of filenames are either both None or one of src/tgt is None.
    # filter out both None since they're not relevant
    cleaned_cache_decompressed_dp = cleaned_cache_decompressed_dp.filter(lambda x: x != (None, None))

    # (None, tgt) => 1, (src, None) => 0
    tgt_data_dp, src_data_dp = cleaned_cache_decompressed_dp.demux(2, lambda x: x.index(None))

    # Pull out the non-None element (i.e., filename) from the tuple
    tgt_data_dp = FileOpener(tgt_data_dp.map(lambda x: x[1]), mode="r")
    src_data_dp = FileOpener(src_data_dp.map(lambda x: x[0]), mode="r")

    src_lines = src_data_dp.readlines(return_path=False, strip_newline=False)
    tgt_lines = tgt_data_dp.readlines(return_path=False, strip_newline=False)
    return src_lines.zip(tgt_lines)
