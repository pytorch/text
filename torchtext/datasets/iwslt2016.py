import os
from functools import partial

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import GDriveReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _clean_files,
    _create_dataset_directory,
    _generate_iwslt_files_for_lang_and_split,
    _wrap_split_argument,
)

URL = "https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8"

_PATH = "2016-01.tgz"

MD5 = "c393ed3fc2a1b0f004b3331043f615ae"

SUPPORTED_DATASETS = {
    "valid_test": ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"],
    "language_pair": {
        "en": ["ar", "de", "fr", "cs"],
        "ar": ["en"],
        "fr": ["en"],
        "de": ["en"],
        "cs": ["en"],
    },
    "year": 16,
}

NUM_LINES = {
    "train": {
        "train": {
            ("ar", "en"): 224126,
            ("de", "en"): 196884,
            ("en", "fr"): 220400,
            ("cs", "en"): 114390,
        }
    },
    "valid": {
        "dev2010": {
            ("ar", "en"): 887,
            ("de", "en"): 887,
            ("en", "fr"): 887,
            ("cs", "en"): 480,
        },
        "tst2010": {
            ("ar", "en"): 1569,
            ("de", "en"): 1565,
            ("en", "fr"): 1664,
            ("cs", "en"): 1511,
        },
        "tst2011": {
            ("ar", "en"): 1199,
            ("de", "en"): 1433,
            ("en", "fr"): 818,
            ("cs", "en"): 1013,
        },
        "tst2012": {
            ("ar", "en"): 1702,
            ("de", "en"): 1700,
            ("en", "fr"): 1124,
            ("cs", "en"): 1385,
        },
        "tst2013": {
            ("ar", "en"): 1169,
            ("de", "en"): 993,
            ("en", "fr"): 1026,
            ("cs", "en"): 1327,
        },
        "tst2014": {("ar", "en"): 1107, ("de", "en"): 1305, ("en", "fr"): 1305},
    },
    "test": {
        "dev2010": {
            ("ar", "en"): 887,
            ("de", "en"): 887,
            ("en", "fr"): 887,
            ("cs", "en"): 480,
        },
        "tst2010": {
            ("ar", "en"): 1569,
            ("de", "en"): 1565,
            ("en", "fr"): 1664,
            ("cs", "en"): 1511,
        },
        "tst2011": {
            ("ar", "en"): 1199,
            ("de", "en"): 1433,
            ("en", "fr"): 818,
            ("cs", "en"): 1013,
        },
        "tst2012": {
            ("ar", "en"): 1702,
            ("de", "en"): 1700,
            ("en", "fr"): 1124,
            ("cs", "en"): 1385,
        },
        "tst2013": {
            ("ar", "en"): 1169,
            ("de", "en"): 993,
            ("en", "fr"): 1026,
            ("cs", "en"): 1327,
        },
        "tst2014": {("ar", "en"): 1107, ("de", "en"): 1305, ("en", "fr"): 1305},
    },
}

SET_NOT_EXISTS = {
    ("en", "ar"): [],
    ("en", "de"): [],
    ("en", "fr"): [],
    ("en", "cs"): ["tst2014"],
    ("ar", "en"): [],
    ("fr", "en"): [],
    ("de", "en"): [],
    ("cs", "en"): ["tst2014"],
}

DATASET_NAME = "IWSLT2016"


def _return_full_filepath(full_filepath, _=None):
    return full_filepath


def _filter_file_name_fn(uncleaned_filename, x):
    return os.path.basename(uncleaned_filename) in x[0]


def _clean_files_wrapper(full_filepath, x):
    return _clean_files(full_filepath, x[0], x[1])


# TODO: migrate this to dataset_utils.py once torchdata is a hard dependency to
# avoid additional conditional imports.
def _filter_clean_cache(cache_decompressed_dp, full_filepath, uncleaned_filename):

    cache_inner_decompressed_dp = cache_decompressed_dp.on_disk_cache(
        filepath_fn=partial(_return_full_filepath, full_filepath)
    )
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.open_files(mode="b").load_from_tar()
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.filter(partial(_filter_file_name_fn, uncleaned_filename))
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.map(partial(_clean_files_wrapper, full_filepath))
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    return cache_inner_decompressed_dp


def _filepath_fn(root, _=None):
    return os.path.join(root, _PATH)


def _inner_iwslt_tar_filepath_fn(inner_iwslt_tar, _=None):
    return inner_iwslt_tar


def _filter_fn(inner_iwslt_tar, x):
    return os.path.basename(inner_iwslt_tar) in x[0]


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def IWSLT2016(
    root=".data",
    split=("train", "valid", "test"),
    language_pair=("de", "en"),
    valid_set="tst2013",
    test_set="tst2014",
):
    """IWSLT2016 dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://wit3.fbk.eu/2016-01

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |"en" |"fr" |"de" |"cs" |"ar" |
    +-----+-----+-----+-----+-----+-----+
    |"en" |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |"fr" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |"de" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |"cs" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+
    |"ar" |  x  |     |     |     |     |
    +-----+-----+-----+-----+-----+-----+

    **valid/test sets**: ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"]


    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language
        valid_set: a string to identify validation set.
        test_set: a string to identify test set.

    :return: DataPipe that yields tuple of source and target sentences
    :rtype: (str, str)

    Examples:
        >>> from torchtext.datasets import IWSLT2016
        >>> train_iter, valid_iter, test_iter = IWSLT2016()
        >>> src_sentence, tgt_sentence = next(iter(train_iter))

    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    if not isinstance(language_pair, list) and not isinstance(language_pair, tuple):
        raise ValueError("language_pair must be list or tuple but got {} instead".format(type(language_pair)))

    assert len(language_pair) == 2, "language_pair must contain only 2 elements: src and tgt language respectively"

    src_language, tgt_language = language_pair[0], language_pair[1]

    if src_language not in SUPPORTED_DATASETS["language_pair"]:
        raise ValueError(
            "src_language '{}' is not valid. Supported source languages are {}".format(
                src_language, list(SUPPORTED_DATASETS["language_pair"])
            )
        )

    if tgt_language not in SUPPORTED_DATASETS["language_pair"][src_language]:
        raise ValueError(
            "tgt_language '{}' is not valid for give src_language '{}'. Supported target language are {}".format(
                tgt_language,
                src_language,
                SUPPORTED_DATASETS["language_pair"][src_language],
            )
        )

    if valid_set not in SUPPORTED_DATASETS["valid_test"] or valid_set in SET_NOT_EXISTS[language_pair]:
        raise ValueError(
            "valid_set '{}' is not valid for given language pair {}. Supported validation sets are {}".format(
                valid_set,
                language_pair,
                [s for s in SUPPORTED_DATASETS["valid_test"] if s not in SET_NOT_EXISTS[language_pair]],
            )
        )

    if test_set not in SUPPORTED_DATASETS["valid_test"] or test_set in SET_NOT_EXISTS[language_pair]:
        raise ValueError(
            "test_set '{}' is not valid for give language pair {}. Supported test sets are {}".format(
                valid_set,
                language_pair,
                [s for s in SUPPORTED_DATASETS["valid_test"] if s not in SET_NOT_EXISTS[language_pair]],
            )
        )

    (file_path_by_lang_and_split, uncleaned_filenames_by_lang_and_split,) = _generate_iwslt_files_for_lang_and_split(
        SUPPORTED_DATASETS["year"], src_language, tgt_language, valid_set, test_set
    )

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp)
    cache_compressed_dp = cache_compressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    languages = "-".join([src_language, tgt_language])

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile. Thus,
    # /root/2016-01/texts/.../src-tgt.tgz will never be in /root/2016-01.tgz/texts/.../src-tgt.tgz
    inner_iwslt_tar = (
        os.path.join(
            root,
            os.path.splitext(_PATH)[0],
            "texts",
            src_language,
            tgt_language,
            languages,
        )
        + ".tgz"
    )

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=partial(_inner_iwslt_tar_filepath_fn, inner_iwslt_tar)
    )
    cache_decompressed_dp = cache_decompressed_dp.open_files(mode="b").load_from_tar()
    cache_decompressed_dp = cache_decompressed_dp.filter(partial(_filter_fn, inner_iwslt_tar))
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    cache_decompressed_dp_1, cache_decompressed_dp_2 = cache_decompressed_dp.fork(num_instances=2)

    src_filename = file_path_by_lang_and_split[src_language][split]
    uncleaned_src_filename = uncleaned_filenames_by_lang_and_split[src_language][split]

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile.
    full_src_filepath = os.path.join(root, "2016-01/texts/", src_language, tgt_language, languages, src_filename)

    cache_inner_src_decompressed_dp = _filter_clean_cache(
        cache_decompressed_dp_1, full_src_filepath, uncleaned_src_filename
    )

    tgt_filename = file_path_by_lang_and_split[tgt_language][split]
    uncleaned_tgt_filename = uncleaned_filenames_by_lang_and_split[tgt_language][split]

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile.
    full_tgt_filepath = os.path.join(root, "2016-01/texts/", src_language, tgt_language, languages, tgt_filename)

    cache_inner_tgt_decompressed_dp = _filter_clean_cache(
        cache_decompressed_dp_2, full_tgt_filepath, uncleaned_tgt_filename
    )

    tgt_data_dp = FileOpener(cache_inner_tgt_decompressed_dp, encoding="utf-8")
    src_data_dp = FileOpener(cache_inner_src_decompressed_dp, encoding="utf-8")

    src_lines = src_data_dp.readlines(return_path=False, strip_newline=False)
    tgt_lines = tgt_data_dp.readlines(return_path=False, strip_newline=False)

    return src_lines.zip(tgt_lines).shuffle().set_shuffle(False).sharding_filter()
