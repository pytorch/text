import os.path
from functools import partial

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import HttpReader
from torchtext.data.datasets_utils import (
    _create_dataset_directory,
)

URL = "http://data.statmt.org/cc-100/%s.txt.xz"

VALID_CODES = {
    "am",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bn",
    "bn_rom",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gn",
    "gu",
    "ha",
    "he",
    "hi",
    "hi_rom",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lg",
    "li",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "my_zaw",
    "ne",
    "nl",
    "no",
    "ns",
    "om",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "qu",
    "rm",
    "ro",
    "ru",
    "sa",
    "si",
    "sc",
    "sd",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "ta_rom",
    "te",
    "te_rom",
    "th",
    "tl",
    "tn",
    "tr",
    "ug",
    "uk",
    "ur",
    "ur_rom",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh-Hans",
    "zh-Hant",
    "zu",
}

NUM_LINES = None
MD5 = None

DATASET_NAME = "CC100"


def _filepath_fn(root, url, _=None):
    return os.path.join(root, os.path.basename(url))


def _decompressed_filepath_fn(root, x):
    return os.path.join(root, os.path.basename(x).rstrip(".xz"))


def _modify_res(language_code, x):
    return language_code, x


@_create_dataset_directory(dataset_name=DATASET_NAME)
def CC100(root: str, language_code: str = "en"):
    """CC100 Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://data.statmt.org/cc-100/

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        language_code: the language of the dataset

    :returns: DataPipe that yields tuple of language code and text
    :rtype: (str, str)
    """
    if language_code not in VALID_CODES:
        raise ValueError(f"Invalid language code {language_code}")

    url = URL % language_code
    url_dp = IterableWrapper([url])
    cache_compressed_dp = url_dp.on_disk_cache(filepath_fn=partial(_filepath_fn, root, url))

    cache_compressed_dp = HttpReader(cache_compressed_dp)
    cache_compressed_dp = cache_compressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_decompressed_filepath_fn, root))
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").load_from_xz()
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb")

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8").readlines(return_path=False)
    return data_dp.map(partial(_modify_res, language_code)).shuffle().set_shuffle(False).sharding_filter()
