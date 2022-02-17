import os
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper


URL = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

MD5 = {
    "train": "b1a00f826fdfbd249f79597b59e1dc12",
    "test": "d52ea96a97a2d943681189a97654912d",
}

NUM_LINES = {
    "train": 120000,
    "test": 7600,
}

DATASET_NAME = "AG_NEWS"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AG_NEWS(root: str, split: Union[Tuple[str], str]):
    """AG_NEWS Dataset

    For additional details refer to https://paperswithcode.com/dataset/ag-news

    Number of lines per split:
        - train: 120000
        - test: 7600

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

    :returns: DataPipe that yields tuple of label (1 to 4) and text
    :rtype: (int, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL[split]])
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, split + ".csv"),
        hash_dict={os.path.join(root, split + ".csv"): MD5[split]},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp)
    cache_dp = cache_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_dp, encoding="utf-8")
    return data_dp.parse_csv().map(fn=lambda t: (int(t[0]), " ".join(t[1:])))
