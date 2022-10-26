import os
from functools import partial
from typing import Union, Tuple

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import HttpReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

URL = {
    "train": "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz",
    "test": "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz",
}

MD5 = {
    "train": "6969c2903a1f19a83569db643e43dcc8",
    "test": "a916e1c2d83eb3004b38fc6fcd628939",
}

NUM_LINES = {
    "train": 8936,
    "test": 2012,
}

_EXTRACTED_FILES = {"train": "train.txt", "test": "test.txt"}

DATASET_NAME = "CoNLL2000Chunking"


def _filepath_fn(root, split, _=None):
    return os.path.join(root, os.path.basename(URL[split]))


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def CoNLL2000Chunking(root: str, split: Union[Tuple[str], str]):
    """CoNLL2000Chunking Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://www.clips.uantwerpen.be/conll2000/chunking/

    Number of lines per split:
        - train: 8936
        - test: 2012

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

    :returns: DataPipe that yields list of words along with corresponding Parts-of-speech tag and chunk tag
    :rtype: [list(str), list(str), list(str)]
    """

    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL[split]])

    # Cache and check HTTP response
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root, split),
        hash_dict={_filepath_fn(root, split): MD5[split]},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    # Cache and check the gzip extraction for relevant split
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").extract(file_type="gzip")
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return data_dp.readlines().read_iob(sep=" ").shuffle().set_shuffle(False).sharding_filter()
