from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper

import os
from torchtext.data.datasets_utils import (
    _clean_files,
    _create_dataset_directory,
    _wrap_split_argument,
)

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


def _filter_clean_cache(cache_decompressed_dp, full_filepath, uncleaned_filename):
    cache_inner_decompressed_dp = cache_decompressed_dp.on_disk_cache(filepath_fn=lambda x: full_filepath)
    cache_inner_decompressed_dp = FileOpener(cache_inner_decompressed_dp, mode="b").read_from_tar()
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.filter(
        lambda x: os.path.basename(uncleaned_filename) in x[0])
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.map(
        lambda x: _clean_files(full_filepath, x[0], x[1]))
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    return cache_inner_decompressed_dp


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

    uncleaned_train_filenames = ('train.tags.{}-{}.{}'.format(src_language, tgt_language, src_language),
                                 'train.tags.{}-{}.{}'.format(src_language, tgt_language, tgt_language))
    uncleaed_valid_filenames = (
    'IWSLT{}.TED.{}.{}-{}.{}.xml'.format(SUPPORTED_DATASETS['year'], valid_set, src_language, tgt_language,
                                         src_language),
    'IWSLT{}.TED.{}.{}-{}.{}.xml'.format(SUPPORTED_DATASETS['year'], valid_set, src_language, tgt_language,
                                         tgt_language))
    uncleaned_test_filenames = (
    'IWSLT{}.TED.{}.{}-{}.{}.xml'.format(SUPPORTED_DATASETS['year'], test_set, src_language, tgt_language,
                                         src_language),
    'IWSLT{}.TED.{}.{}-{}.{}.xml'.format(SUPPORTED_DATASETS['year'], test_set, src_language, tgt_language,
                                         tgt_language))

    uncleaned_src_train, uncleaned_tgt_train = uncleaned_train_filenames
    uncleaned_src_eval, uncleaned_tgt_eval = uncleaed_valid_filenames
    uncleaned_src_test, uncleaned_tgt_test = uncleaned_test_filenames

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH),
        hash_dict={os.path.join(root, _PATH): MD5},
        hash_type="md5"
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp)
    cache_compressed_dp = cache_compressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    languages = "-".join([src_language, tgt_language])

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile. Thus,
    # /root/2017-01-trnmted/texts/.../src-tgt.tgz will never be in
    # /root/2017-01-trnmted.tgz/texts/.../src-tgt.tgz
    inner_iwslt_tar = os.path.join(
        root,
        os.path.splitext(_PATH)[0],
        "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz"
    )

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=lambda x: inner_iwslt_tar)
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").read_from_tar().filter(
        lambda x: os.path.basename(inner_iwslt_tar) in x[0])
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    file_path_by_lang_and_split = {
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
    }

    uncleaned_filenames = {
        src_language: {
            "train": uncleaned_src_train,
            "valid": uncleaned_src_eval,
            "test": uncleaned_src_test,
        },
        tgt_language: {
            "train": uncleaned_tgt_train,
            "valid": uncleaned_tgt_eval,
            "test": uncleaned_tgt_test,
        }
    }

    src_filename = file_path_by_lang_and_split[src_language][split]
    uncleaned_src_filename = uncleaned_filenames[src_language][split]

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile.
    full_src_filepath = os.path.join(root, "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo", src_filename)

    cache_inner_src_decompressed_dp = _filter_clean_cache(cache_decompressed_dp, full_src_filepath,
                                                          uncleaned_src_filename)

    tgt_filename = file_path_by_lang_and_split[tgt_language][split]
    uncleaned_tgt_filename = uncleaned_filenames[tgt_language][split]

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile.
    full_tgt_filepath = os.path.join(root, "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo", tgt_filename)

    cache_inner_tgt_decompressed_dp = _filter_clean_cache(cache_decompressed_dp, full_tgt_filepath,
                                                          uncleaned_tgt_filename)

    tgt_data_dp = FileOpener(cache_inner_tgt_decompressed_dp, mode="r")
    src_data_dp = FileOpener(cache_inner_src_decompressed_dp, mode="r")

    src_lines = src_data_dp.readlines(return_path=False, strip_newline=False)
    tgt_lines = tgt_data_dp.readlines(return_path=False, strip_newline=False)

    return src_lines.zip(tgt_lines)
