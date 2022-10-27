import csv
import os
from functools import partial
from typing import Union, Tuple

from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)


URL = {
    "train": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
    "test": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt",
}

MD5 = {
    "train": "793daf7b6224281e75fe61c1f80afe35",
    "test": "e437fdddb92535b820fe8852e2df8a49",
}

NUM_LINES = {
    "train": 4076,
    "test": 1725,
}


DATASET_NAME = "MRPC"


def _filepath_fn(root, x):
    return os.path.join(root, os.path.basename(x))


def _modify_res(x):
    return (int(x[0]), x[3], x[4])


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def MRPC(root: str, split: Union[Tuple[str], str]):
    """MRPC Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://www.microsoft.com/en-us/download/details.aspx?id=52398

    Number of lines per split:
        - train: 4076
        - test: 1725

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

    :returns: DataPipe that yields data points from MRPC dataset which consist of label, sentence1, sentence2
    :rtype: (int, str, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL[split]])
    # cache data on-disk with sanity check
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root, URL[split]): MD5[split]},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, encoding="utf-8")
    cache_dp = cache_dp.parse_csv(skip_lines=1, delimiter="\t", quoting=csv.QUOTE_NONE).map(_modify_res)
    return cache_dp.shuffle().set_shuffle(False).sharding_filter()
