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

_EXTRACTED_FILES_MD5 = {
    'train': "2e2f24e90e20fcb910ab2251b5ed8cd0",
    'test': "56944df34be553b72a2a634e539a0951"
}


DATASET_NAME = "CoNLL2000Chunking"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def CoNLL2000Chunking(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")

    url_dp = IterableWrapper([URL[split]])

    # Cache and check HTTP response
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, "conll2000chunking", os.path.basename(URL[split])),
        hash_dict={os.path.join(root, "conll2000chunking", os.path.basename(URL[split])): MD5[split]},
        hash_type="md5"
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, mode="b")

    # Cache and check the gzip extraction for relevant split
    cache_dp = cache_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, "conll2000chunking", _EXTRACTED_FILES[split]),
        hash_dict={os.path.join(root, "conll2000chunking", _EXTRACTED_FILES[split]): _EXTRACTED_FILES_MD5[split]},
        hash_type="md5"
    )
    cache_dp = cache_dp.extract(file_type="gzip").filter(lambda x: _EXTRACTED_FILES[split] in x[0])
    cache_dp = cache_dp.end_caching(mode="wb")

    cache_dp = FileOpener(cache_dp, mode="b")
    return cache_dp.readlines(decode=True).read_iob(sep=" ")
