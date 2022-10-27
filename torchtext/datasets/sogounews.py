import os
from functools import partial
from typing import Union, Tuple

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import GDriveReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE"

MD5 = "0c1700ba70b73f964dd8de569d3fd03e"

NUM_LINES = {
    "train": 450000,
    "test": 60000,
}

_PATH = "sogou_news_csv.tar.gz"

_EXTRACTED_FILES = {
    "train": os.path.join("sogou_news_csv", "train.csv"),
    "test": os.path.join("sogou_news_csv", "test.csv"),
}

_EXTRACTED_FILES_MD5 = {
    "train": "f36156164e6eac2feda0e30ad857eef0",
    "test": "59e493c41cee050329446d8c45615b38",
}

DATASET_NAME = "SogouNews"


def _filepath_fn(root, _=None):
    return os.path.join(root, _PATH)


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, x):
    return _EXTRACTED_FILES[split] in x[0]


def _modify_res(t):
    return int(t[0]), " ".join(t[1:])


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def SogouNews(root: str, split: Union[Tuple[str], str]):
    """SogouNews Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://arxiv.org/abs/1509.01626

     Number of lines per split:
         - train: 450000
         - test: 60000

     Args:
         root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
         split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

     :returns: DataPipe that yields tuple of label (1 to 5) and text containing the news title and contents
     :rtype: (int, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_tar().filter(partial(_filter_fn, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return data_dp.parse_csv().map(fn=_modify_res).shuffle().set_shuffle(False).sharding_filter()
