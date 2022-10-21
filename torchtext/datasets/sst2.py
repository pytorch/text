# Copyright (c) Facebook, Inc. and its affiliates.
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


URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

MD5 = "9f81648d4199384278b86e315dac217c"

NUM_LINES = {
    "train": 67349,
    "dev": 872,
    "test": 1821,
}

_PATH = "SST-2.zip"

DATASET_NAME = "SST2"

_EXTRACTED_FILES = {
    "train": os.path.join("SST-2", "train.tsv"),
    "dev": os.path.join("SST-2", "dev.tsv"),
    "test": os.path.join("SST-2", "test.tsv"),
}


def _filepath_fn(root, _=None):
    return os.path.join(root, os.path.basename(URL))


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, x):
    return _EXTRACTED_FILES[split] in x[0]


def _modify_test_res(t):
    return (t[1].strip(),)


def _modify_res(t):
    return t[0].strip(), int(t[1])


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev", "test"))
def SST2(root, split):
    """SST2 Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://nlp.stanford.edu/sentiment/

    Number of lines per split:
        - train: 67349
        - dev: 872
        - test: 1821

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev`, `test`)

    :returns: DataPipe that yields tuple of text and/or label (1 to 4). The `test` split only returns text.
    :rtype: Union[(int, str), (str,)]
    """
    # TODO Remove this after removing conditional dependency
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
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_zip().filter(partial(_filter_fn, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    # test split for SST2 doesn't have labels
    if split == "test":
        parsed_data = data_dp.parse_csv(skip_lines=1, delimiter="\t").map(_modify_test_res)
    else:
        parsed_data = data_dp.parse_csv(skip_lines=1, delimiter="\t").map(_modify_res)
    return parsed_data.shuffle().set_shuffle(False).sharding_filter()
