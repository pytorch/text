# Copyright (c) Facebook, Inc. and its affiliates.
import os
import csv

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _create_dataset_directory,
    _wrap_split_argument,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper

    # we import HttpReader from _download_hooks so we can swap out public URLs
    # with interal URLs when the dataset is used within Facebook
    from torchtext._download_hooks import HttpReader


URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"

MD5 = "0f70aaf66293b3c088a864891db51353"

NUM_LINES = {
    "train": 392702,
    "dev": 9714,
    "dev_mismatched": 9832,
}

_PATH = "multinli_1.0.zip"

DATASET_NAME = "MNLI"

_EXTRACTED_FILES = {
    "train": "multinli_1.0_train.txt",
    "dev_matched": "multinli_1.0_dev_matched.txt",
    "dev_mismatched": "multinli_1.0_dev_mismatched.txt",
}

LABEL_TO_INT = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev_matched", "dev_mismatched"))
def MNLI(root, split):
    """MNLI Dataset

    For additional details refer to https://cims.nyu.edu/~sbowman/multinli/

    Number of lines per split:
        - train: 392702
        - dev_matched: 9714
        - dev_mismatched: 9832

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev_matched`, `dev_mismatched`)

    :returns: DataPipe that yields tuple of text and/or label (0 to 2).
    :rtype: Tuple[int, str, str]
    """
    # TODO Remove this after removing conditional dependency
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(x)),
        hash_dict={os.path.join(root, os.path.basename(URL)): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _EXTRACTED_FILES[split])
    )
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").read_from_zip().filter(lambda x: _EXTRACTED_FILES[split] in x[0])
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    parsed_data = data_dp.parse_csv(skip_lines=1, delimiter="\t", quoting=csv.QUOTE_NONE).filter(lambda x: x[0] in LABEL_TO_INT).map(lambda x: (LABEL_TO_INT[x[0]], x[5], x[6]))
    return parsed_data
