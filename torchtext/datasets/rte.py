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


URL = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip"

MD5 = "bef554d0cafd4ab6743488101c638539"

NUM_LINES = {
    "train": 2490,
    "dev": 277,
    "test": 3000,
}

_PATH = "RTE.zip"

DATASET_NAME = "RTE"

_EXTRACTED_FILES = {
    "train": os.path.join("RTE", "train.tsv"),
    "dev": os.path.join("RTE", "dev.tsv"),
    "test": os.path.join("RTE", "test.tsv"),
}

MAP_LABELS = {"entailment": 0, "not_entailment": 1}


def _filepath_fn(root, x=None):
    return os.path.join(root, os.path.basename(x))


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, x):
    return _EXTRACTED_FILES[split] in x[0]


def _modify_res(split, x):
    if split == "test":
        return (x[1], x[2])
    else:
        return (MAP_LABELS[x[3]], x[1], x[2])


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev", "test"))
def RTE(root, split):
    """RTE Dataset

    For additional details refer to https://aclweb.org/aclwiki/Recognizing_Textual_Entailment

    Number of lines per split:
        - train: 2490
        - dev: 277
        - test: 3000

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev`, `test`)

    :returns: DataPipe that yields tuple of text and/or label (0 and 1). The `test` split only returns text.
    :rtype: Union[(int, str, str), (str, str)]
    """
    # TODO Remove this after removing conditional dependency
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
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
    parsed_data = data_dp.parse_csv(skip_lines=1, delimiter="\t", quoting=csv.QUOTE_NONE).map(
        partial(_modify_res, split)
    )
    return parsed_data.shuffle().set_shuffle(False).sharding_filter()
