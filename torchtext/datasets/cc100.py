import os.path

from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper

from torchtext.data.datasets_utils import (
    _create_dataset_directory,
)

URL = "http://data.statmt.org/cc-100/%s.txt.xz"

VALID_CODES = {
    "am", "ar", "as", "az", "be", "bg", "bn", "bn_rom", "br", "bs", "ca", "cs", "cy", "da", "de",
    "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gn", "gu",
    "ha", "he", "hi", "hi_rom", "hr", "ht", "hu", "hy", "id", "ig", "is", "it", "ja", "jv", "ka",
    "kk", "km", "kn", "ko", "ku", "ky", "la", "lg", "li", "ln", "lo", "lt", "lv", "mg", "mk", "ml",
    "mn", "mr", "ms", "my", "my_zaw", "ne", "nl", "no", "ns", "om", "or", "pa", "pl", "ps", "pt",
    "qu", "rm", "ro", "ru", "sa", "si", "sc", "sd", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv",
    "sw", "ta", "ta_rom", "te", "te_rom", "th", "tl", "tn", "tr", "ug", "uk", "ur", "ur_rom", "uz",
    "vi", "wo", "xh", "yi", "yo", "zh-Hans", "zh-Hant", "zu",
}

NUM_LINES = None
MD5 = None

DATASET_NAME = "CC100"


@_create_dataset_directory(dataset_name=DATASET_NAME)
def CC100(root: str, language_code: str):
    if language_code not in VALID_CODES:
        raise ValueError(f"Invalid language code {language_code}")

    url = URL % language_code
    url_dp = IterableWrapper([url])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(url))
    )

    cache_compressed_dp = HttpReader(cache_compressed_dp)
    cache_compressed_dp = cache_compressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    cache_compressed_dp = FileOpener(cache_compressed_dp, mode="b").map(lambda x: x[0])

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(x).rstrip(".xz"))
    )
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").read_from_xz()
    cache_decompressed_dp = cache_decompressed_dp.filter(lambda x: os.path.basename(x).rstrip(".xz") in x[0])
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, mode="r")

    units_dp = data_dp.readlines().map(lambda x: (language_code, x[1]))
    return units_dp
