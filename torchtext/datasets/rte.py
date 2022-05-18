# Copyright (c) Facebook, Inc. and its affiliates.
import os

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


URL = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip"

MD5 = "bef554d0cafd4ab6743488101c638539"

NUM_LINES = {
    "train": 67349,
    "dev": 872,
    "test": 1821,
}

_PATH = "RTE.zip"

DATASET_NAME = "RTE"

_EXTRACTED_FILES = {
    "train": os.path.join("RTE", "train.tsv"),
    "dev": os.path.join("RTE", "dev.tsv"),
    "test": os.path.join("RTE", "test.tsv"),
}


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev", "test"))
def RTE(root, split):
    """RTE Dataset

    For additional details refer to https://aclweb.org/aclwiki/Recognizing_Textual_Entailment

    Number of lines per split:
        - train: 67349
        - dev: 872
        - test: 1821

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

    def _filepath_fn(x=None):
        return os.path.join(root, os.path.basename(x))

    def _extracted_filepath_fn(_=None):
        return os.path.join(root, _EXTRACTED_FILES[split])

    def _filter_fn(x):
        return _EXTRACTED_FILES[split] in x[0]

    def _modify_res(x):
        if split == 'test':
            return (x[1], x[2])
        else:
            return (int(x[3]), x[1], x[2])

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=_filepath_fn,
        hash_dict={_filepath_fn(URL): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=_extracted_filepath_fn
    )
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").read_from_zip().filter(_filter_fn)
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    parsed_data = data_dp.parse_csv(skip_lines=1, delimiter="\t").map(_modify_res)
    return parsed_data.shuffle().set_shuffle(False).sharding_filter()
