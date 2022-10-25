# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import os
from functools import partial

from torchdata.datapipes.iter import FileOpener, IterableWrapper

# we import HttpReader from _download_hooks so we can swap out public URLs
# with interal URLs when the dataset is used within Facebook
from torchtext._download_hooks import HttpReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _create_dataset_directory,
    _wrap_split_argument,
)


URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"

MD5 = "0f70aaf66293b3c088a864891db51353"

NUM_LINES = {
    "train": 392702,
    "dev_matched": 9815,
    "dev_mismatched": 9832,
}

_PATH = "multinli_1.0.zip"

DATASET_NAME = "MNLI"

_EXTRACTED_FILES = {
    "train": "multinli_1.0_train.txt",
    "dev_matched": "multinli_1.0_dev_matched.txt",
    "dev_mismatched": "multinli_1.0_dev_mismatched.txt",
}

LABEL_TO_INT = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _filepath_fn(root, x=None):
    return os.path.join(root, os.path.basename(x))


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, x):
    return _EXTRACTED_FILES[split] in x[0]


def _filter_res(x):
    return x[0] in LABEL_TO_INT


def _modify_res(x):
    return (LABEL_TO_INT[x[0]], x[5], x[6])


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev_matched", "dev_mismatched"))
def MNLI(root, split):
    """MNLI Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://cims.nyu.edu/~sbowman/multinli/

    Number of lines per split:
        - train: 392702
        - dev_matched: 9815
        - dev_mismatched: 9832

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev_matched`, `dev_mismatched`)

    :returns: DataPipe that yields tuple of text and label (0 to 2).
    :rtype: Tuple[int, str, str]
    """
    # TODO Remove this after removing conditional dependency
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root, URL): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_zip().filter(partial(_filter_fn, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    parsed_data = (
        data_dp.parse_csv(skip_lines=1, delimiter="\t", quoting=csv.QUOTE_NONE).filter(_filter_res).map(_modify_res)
    )
    return parsed_data.shuffle().set_shuffle(False).sharding_filter()
