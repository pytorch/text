from typing import Tuple, Union

from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper

import os

from torchtext.data.datasets_utils import _add_docstring_header, _create_dataset_directory, _wrap_split_argument

URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU"

MD5 = "f3f9899b997a42beb24157e62e3eea8d"

NUM_LINES = {
    "train": 1400000,
    "test": 60000,
}

_PATH = "yahoo_answers_csv.tar.gz"

DATASET_NAME = "YahooAnswers"

_EXTRACTED_FILES = {
    "train": os.path.join("yahoo_answers_csv", "train.csv"),
    "test": os.path.join("yahoo_answers_csv", "test.csv"),
}


@_add_docstring_header(num_lines=NUM_LINES, num_classes=10)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def YahooAnswers(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])

    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH), hash_dict={os.path.join(root, _PATH): MD5}, hash_type="md5"
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _EXTRACTED_FILES[split])
    )
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b")
    cache_decompressed_dp = cache_decompressed_dp.read_from_tar()
    cache_decompressed_dp = cache_decompressed_dp.filter(lambda x: _EXTRACTED_FILES[split] in x[0])
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, mode="b")

    return data_dp.parse_csv().map(fn=lambda t: (int(t[0]), " ".join(t[1:])))
