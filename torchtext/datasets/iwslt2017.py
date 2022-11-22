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

URL = "https://drive.google.com/u/0/uc?id=12ycYSzLIG253AFN35Y6qoyf9wtkOjakp"
_PATH = "2017-01-trnmted.tgz"
MD5 = "aca701032b1c4411afc4d9fa367796ba"

SUPPORTED_DATASETS = {
    "valid_test": ["dev2010", "tst2010"],
    "language_pair": {
        "en": ["nl", "de", "it", "ro"],
        "ro": ["de", "en", "nl", "it"],
        "de": ["ro", "en", "nl", "it"],
        "it": ["en", "nl", "de", "ro"],
        "nl": ["de", "en", "it", "ro"],
    },
    "year": 17,
}

NUM_LINES = {
    "train": {
        "train": {
            ("en", "nl"): 237240,
            ("de", "en"): 206112,
            ("en", "it"): 231619,
            ("en", "ro"): 220538,
            ("de", "ro"): 201455,
            ("nl", "ro"): 206920,
            ("it", "ro"): 217551,
            ("de", "nl"): 213628,
            ("de", "it"): 205465,
            ("it", "nl"): 233415,
        }
    },
    "valid": {
        "dev2010": {
            ("en", "nl"): 1003,
            ("de", "en"): 888,
            ("en", "it"): 929,
            ("en", "ro"): 914,
            ("de", "ro"): 912,
            ("nl", "ro"): 913,
            ("it", "ro"): 914,
            ("de", "nl"): 1001,
            ("de", "it"): 923,
            ("it", "nl"): 1001,
        },
        "tst2010": {
            ("en", "nl"): 1777,
            ("de", "en"): 1568,
            ("en", "it"): 1566,
            ("en", "ro"): 1678,
            ("de", "ro"): 1677,
            ("nl", "ro"): 1680,
            ("it", "ro"): 1643,
            ("de", "nl"): 1779,
            ("de", "it"): 1567,
            ("it", "nl"): 1669,
        },
    },
    "test": {
        "dev2010": {
            ("en", "nl"): 1003,
            ("de", "en"): 888,
            ("en", "it"): 929,
            ("en", "ro"): 914,
            ("de", "ro"): 912,
            ("nl", "ro"): 913,
            ("it", "ro"): 914,
            ("de", "nl"): 1001,
            ("de", "it"): 923,
            ("it", "nl"): 1001,
        },
        "tst2010": {
            ("en", "nl"): 1777,
            ("de", "en"): 1568,
            ("en", "it"): 1566,
            ("en", "ro"): 1678,
            ("de", "ro"): 1677,
            ("nl", "ro"): 1680,
            ("it", "ro"): 1643,
            ("de", "nl"): 1779,
            ("de", "it"): 1567,
            ("it", "nl"): 1669,
        },
    },
}

DATASET_NAME = "IWSLT2017"


def _return_full_filepath(full_filepath, _=None):
    return full_filepath


def _filter_filename_fn(uncleaned_filename, x):
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
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.filter(partial(_filter_filename_fn, uncleaned_filename))
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.map(partial(_clean_files_wrapper, full_filepath))
    cache_inner_decompressed_dp = cache_inner_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    return cache_inner_decompressed_dp


def _filepath_fn(root, _=None):
    return os.path.join(root, _PATH)


def _inner_iwslt_tar_filepath_fn(inner_iwslt_tar, _=None):
    return inner_iwslt_tar


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def IWSLT2017(root=".data", split=("train", "valid", "test"), language_pair=("de", "en")):
    """IWSLT2017 dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://wit3.fbk.eu/2017-01

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+-----+-----+-----+
    |     |"en" |"nl" |"de" |"it" |"ro" |
    +-----+-----+-----+-----+-----+-----+
    |"en" |     |   x |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |"nl" |  x  |     |  x  |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |"de" |  x  |   x |     |  x  |  x  |
    +-----+-----+-----+-----+-----+-----+
    |"it" |  x  |   x |  x  |     |  x  |
    +-----+-----+-----+-----+-----+-----+
    |"ro" |  x  |   x |  x  |  x  |     |
    +-----+-----+-----+-----+-----+-----+


    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language

    :return: DataPipe that yields tuple of source and target sentences
    :rtype: (str, str)

    Examples:
        >>> from torchtext.datasets import IWSLT2017
        >>> train_iter, valid_iter, test_iter = IWSLT2017()
        >>> src_sentence, tgt_sentence = next(iter(train_iter))

    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    valid_set = "dev2010"
    test_set = "tst2010"

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

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile. Thus,
    # /root/2017-01-trnmted/texts/.../DeEnItNlRo-DeEnItNlRo.tgz will never be in
    # /root/2017-01-trnmted.tgz/texts/.../DeEnItNlRo-DeEnItNlRo.tgz
    inner_iwslt_tar = os.path.join(
        root,
        os.path.splitext(_PATH)[0],
        "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz",
    )

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=partial(_inner_iwslt_tar_filepath_fn, inner_iwslt_tar)
    )
    cache_decompressed_dp = cache_decompressed_dp.open_files(mode="b").load_from_tar()
    # As we had filenames duplicated, any trash files in archive can become tgz

    def extracted_file_name(inner_iwslt_tar, inner_tar_name):
        name = os.path.basename(inner_tar_name)
        path = os.path.dirname(inner_iwslt_tar)
        return os.path.join(path, name)

    cache_decompressed_dp = cache_decompressed_dp.end_caching(
        mode="wb", filepath_fn=partial(extracted_file_name, inner_iwslt_tar)
    )
    # As we corrected path, we need to leave tgz files only now and no dot files

    def leave_only_tgz(file_name):
        name = os.path.basename(file_name)
        _, file_extension = os.path.splitext(file_name)
        return file_extension == ".tgz" and name[0] != "."

    cache_decompressed_dp = cache_decompressed_dp.filter(leave_only_tgz)
    cache_decompressed_dp_1, cache_decompressed_dp_2 = cache_decompressed_dp.fork(num_instances=2)

    src_filename = file_path_by_lang_and_split[src_language][split]
    uncleaned_src_filename = uncleaned_filenames_by_lang_and_split[src_language][split]

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile.
    full_src_filepath = os.path.join(
        root,
        os.path.splitext(_PATH)[0],
        "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo",
        src_filename,
    )

    cache_inner_src_decompressed_dp = _filter_clean_cache(
        cache_decompressed_dp_1, full_src_filepath, uncleaned_src_filename
    )

    tgt_filename = file_path_by_lang_and_split[tgt_language][split]
    uncleaned_tgt_filename = uncleaned_filenames_by_lang_and_split[tgt_language][split]

    # We create the whole filepath here, but only check for the literal filename in the filter
    # because we're lazily extracting from the outer tarfile.
    full_tgt_filepath = os.path.join(
        root,
        os.path.splitext(_PATH)[0],
        "texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo",
        tgt_filename,
    )

    cache_inner_tgt_decompressed_dp = _filter_clean_cache(
        cache_decompressed_dp_2, full_tgt_filepath, uncleaned_tgt_filename
    )

    tgt_data_dp = FileOpener(cache_inner_tgt_decompressed_dp, encoding="utf-8")
    src_data_dp = FileOpener(cache_inner_src_decompressed_dp, encoding="utf-8")

    src_lines = src_data_dp.readlines(return_path=False, strip_newline=False)
    tgt_lines = tgt_data_dp.readlines(return_path=False, strip_newline=False)

    return src_lines.zip(tgt_lines).shuffle().set_shuffle(False).sharding_filter()
