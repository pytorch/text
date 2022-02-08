from torchtext._internal.module_utils import is_module_available
from typing import Union, Tuple

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper

from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

import os

URL = {
    'train': "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz",
    'test': "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz",
}

MD5 = {
    'train': "6969c2903a1f19a83569db643e43dcc8",
    'test': "a916e1c2d83eb3004b38fc6fcd628939",
}

NUM_LINES = {
    'train': 8936,
    'test': 2012,
}

_EXTRACTED_FILES = {
    'train': 'train.txt',
    'test': 'test.txt'
}

DATASET_NAME = "CoNLL2000Chunking"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def CoNLL2000Chunking(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")

    url_dp = IterableWrapper([URL[split]])

    # Cache and check HTTP response
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(URL[split])),
        hash_dict={os.path.join(root, os.path.basename(URL[split])): MD5[split]},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(
        mode="wb", same_filepath_fn=True
    )

    # Cache and check the gzip extraction for relevant split
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _EXTRACTED_FILES[split])
    )
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").extract(
        file_type="gzip"
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(
        mode="wb", same_filepath_fn=True
    )

    data_dp = FileOpener(cache_decompressed_dp, mode="b")
    return data_dp.readlines(decode=True).read_iob(sep=" ")
