import os
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper


URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA"

MD5 = "57d28bd5d930e772930baddf36641c7c"

NUM_LINES = {
    "train": 3000000,
    "test": 650000,
}

_PATH = "amazon_review_full_csv.tar.gz"

_EXTRACTED_FILES = {
    "train": os.path.join("amazon_review_full_csv", "train.csv"),
    "test": os.path.join("amazon_review_full_csv", "test.csv"),
}

_EXTRACTED_FILES_MD5 = {
    "train": "31b268b09fd794e0ca5a1f59a0358677",
    "test": "0f1e78ab60f625f2a30eab6810ef987c",
}

DATASET_NAME = "AmazonReviewFull"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=5)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AmazonReviewFull(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH),
        hash_dict={os.path.join(root, _PATH): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(
        mode="wb", same_filepath_fn=True
    )

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _EXTRACTED_FILES[split])
    )
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b")
        .read_from_tar()
        .filter(lambda x: _EXTRACTED_FILES[split] in x[0])
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(
        mode="wb", same_filepath_fn=True
    )

    data_dp = FileOpener(cache_decompressed_dp, mode="b")
    return data_dp.parse_csv().map(fn=lambda t: (int(t[0]), " ".join(t[1:])))
