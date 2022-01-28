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

URL = 'https://drive.google.com/u/0/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp'
_PATH = '2017-01-trnmted.tgz'
MD5 = 'aca701032b1c4411afc4d9fa367796ba'

SUPPORTED_DATASETS = {
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

NUM_LINES = {
    'train': {
        'train': {
            ('en', 'nl'): 237240,
            ('de', 'en'): 206112,
            ('en', 'it'): 231619,
            ('en', 'ro'): 220538,
            ('de', 'ro'): 201455,
            ('nl', 'ro'): 206920,
            ('it', 'ro'): 217551,
            ('de', 'nl'): 213628,
            ('de', 'it'): 205465,
            ('it', 'nl'): 233415
        }
    },
    'valid': {
        'dev2010': {
            ('en', 'nl'): 1003,
            ('de', 'en'): 888,
            ('en', 'it'): 929,
            ('en', 'ro'): 914,
            ('de', 'ro'): 912,
            ('nl', 'ro'): 913,
            ('it', 'ro'): 914,
            ('de', 'nl'): 1001,
            ('de', 'it'): 923,
            ('it', 'nl'): 1001
        },
        'tst2010': {
            ('en', 'nl'): 1777,
            ('de', 'en'): 1568,
            ('en', 'it'): 1566,
            ('en', 'ro'): 1678,
            ('de', 'ro'): 1677,
            ('nl', 'ro'): 1680,
            ('it', 'ro'): 1643,
            ('de', 'nl'): 1779,
            ('de', 'it'): 1567,
            ('it', 'nl'): 1669
        }
    },
    'test': {
        'dev2010': {
            ('en', 'nl'): 1003,
            ('de', 'en'): 888,
            ('en', 'it'): 929,
            ('en', 'ro'): 914,
            ('de', 'ro'): 912,
            ('nl', 'ro'): 913,
            ('it', 'ro'): 914,
            ('de', 'nl'): 1001,
            ('de', 'it'): 923,
            ('it', 'nl'): 1001
        },
        'tst2010': {
            ('en', 'nl'): 1777,
            ('de', 'en'): 1568,
            ('en', 'it'): 1566,
            ('en', 'ro'): 1678,
            ('de', 'ro'): 1677,
            ('nl', 'ro'): 1680,
            ('it', 'ro'): 1643,
            ('de', 'nl'): 1779,
            ('de', 'it'): 1567,
            ('it', 'nl'): 1669
        }
    }
}

DATASET_NAME = "IWSLT2017"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def IWSLT2017(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en')):
    """IWSLT2017 dataset

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |'en' |'nl' |'de' |'it' |'ro' |
    +-----+-----+-----+-----+-----+-----+
    |'en' |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'nl' |  x  |     |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'de' |  x  |   x |     |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'it' |  x  |   x |  x  |     |  x  |
    +-----+-----+-----+-----+-----+-----+
    |'ro' |  x  |   x |  x  |  x  |     |
    +-----+-----+-----+-----+-----+-----+


    For additional details refer to source website: https://wit3.fbk.eu/2017-01

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language

    Examples:
        >>> from torchtext.datasets import IWSLT2017
        >>> train_iter, valid_iter, test_iter = IWSLT2017()
        >>> src_sentence, tgt_sentence = next(train_iter)

    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")

    valid_set = 'dev2010'
    test_set = 'tst2010'

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

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        # Convert /path/to/downloaded/foo.tgz to /path/to/downloaded/foo/rest/of/path
        filepath_fn=lambda x: os.path.join(root, os.path.splitext(_PATH)[0], "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz")
    )
    cache_decompressed_dp = cache_decompressed_dp.read_from_tar()
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    def clean_files(fname):
        if 'xml' in fname:
            _clean_xml_file(fname)
            return os.path.splitext(fname)[0]
        elif "tags" in fname:
            _clean_tags_file(fname)
            return fname.replace('.tags', '')
        return fname

    cache_decompressed_dp = cache_decompressed_dp.on_disk_cache(
        # Convert /path/to/downloaded/foo.tgz to /path/to/downloaded/foo/rest/of/path
        filepath_fn=lambda x: os.path.join(root, os.path.splitext(_PATH)[0], "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo/")
    )

    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").read_from_tar()
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=False)

    def get_filepath(split, lang):
        return {
            src_language: {
                "train": src_train,
                "valid": src_eval,
                "test": src_test,
            },
            tgt_language: {
                "train": tgt_train,
                "valid": tgt_eval,
                "test": tgt_test,
            }
        }[lang][split]

    cache_decompressed_dp = cache_decompressed_dp.flatmap(FileLister)
    cleaned_cache_decompressed_dp = cache_decompressed_dp.map(clean_files)

    # Filters out irrelevant file given the filename templates filled with split and src/tgt codes
    src_data_dp = cleaned_cache_decompressed_dp.filter(lambda x: get_filepath(split, src_language) in x)
    tgt_data_dp = cleaned_cache_decompressed_dp.filter(lambda x: get_filepath(split, tgt_language) in x)

    tgt_data_dp = FileOpener(tgt_data_dp, mode="r")
    src_data_dp = FileOpener(src_data_dp, mode="r")

    src_lines = src_data_dp.readlines(return_path=False, strip_newline=False)
    tgt_lines = tgt_data_dp.readlines(return_path=False, strip_newline=False)
    return src_lines.zip(tgt_lines)
