import csv
import os
from functools import partial
from typing import Union, Tuple

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import HttpReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _create_dataset_directory, _wrap_split_argument

URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"

MD5 = "9f6d88c3558ec424cd9d66ea03589aba"

_PATH = "cola_public_1.1.zip"

NUM_LINES = {"train": 8551, "dev": 527, "test": 516}

_EXTRACTED_FILES = {
    "train": os.path.join("cola_public", "raw", "in_domain_train.tsv"),
    "dev": os.path.join("cola_public", "raw", "in_domain_dev.tsv"),
    "test": os.path.join("cola_public", "raw", "out_of_domain_dev.tsv"),
}

DATASET_NAME = "CoLA"


def _filepath_fn(root, _=None):
    return os.path.join(root, _PATH)


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, x):
    return _EXTRACTED_FILES[split] in x[0]


def _modify_res(t):
    return (t[0], int(t[1]), t[3])


def _filter_res(x):
    return len(x) == 4


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev", "test"))
def CoLA(root: str, split: Union[Tuple[str], str]):
    """CoLA dataset

    .. warning::

        Using datapipes is still currently subject to a few caveats. If you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://nyu-mll.github.io/CoLA/

    Number of lines per split:
        - train: 8551
        - dev: 527
        - test: 516

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev`, `test`)


    :returns: DataPipe that yields rows from CoLA dataset (source (str), label (int), sentence (str))
    :rtype: (str, int, str)
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
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_zip().filter(partial(_filter_fn, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    # some context stored at top of the file needs to be removed
    parsed_data = (
        data_dp.parse_csv(skip_lines=1, delimiter="\t", quoting=csv.QUOTE_NONE).filter(_filter_res).map(_modify_res)
    )
    return parsed_data.shuffle().set_shuffle(False).sharding_filter()
