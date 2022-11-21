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
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MD5 = {
    "train": "981b29407e0affa3b1b156f72073b945",
    "dev": "3e85deb501d4e538b6bc56f786231552",
}

NUM_LINES = {
    "train": 87599,
    "dev": 10570,
}


DATASET_NAME = "SQuAD1"


def _filepath_fn(root, split, _=None):
    return os.path.join(root, os.path.basename(URL[split]))


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev"))
def SQuAD1(root: str, split: Union[Tuple[str], str]):
    """SQuAD1 Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://rajpurkar.github.io/SQuAD-explorer/

    Number of lines per split:
        - train: 87599
        - dev: 10570

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev`)

    :returns: DataPipe that yields data points from SQuaAD1 dataset which consist of context, question, list of answers and corresponding index in context
    :rtype: (str, str, list(str), list(int))
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL[split]])
    # cache data on-disk with sanity check
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root, split),
        hash_dict={_filepath_fn(root, split): MD5[split]},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, encoding="utf-8")
    return cache_dp.parse_json_files().read_squad().shuffle().set_shuffle(False).sharding_filter()
