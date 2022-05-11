import os
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper
    from torchtext._download_hooks import GDriveReader

URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg"

MD5 = "620c8ae4bd5a150b730f1ba9a7c6a4d3"

NUM_LINES = {
    "train": 560000,
    "test": 38000,
}

_PATH = "yelp_review_polarity_csv.tar.gz"

DATASET_NAME = "YelpReviewPolarity"

_EXTRACTED_FILES = {
    "train": os.path.join("yelp_review_polarity_csv", "train.csv"),
    "test": os.path.join("yelp_review_polarity_csv", "test.csv"),
}


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def YelpReviewPolarity(root: str, split: Union[Tuple[str], str]):
    """YelpReviewPolarity Dataset

    For additional details refer to https://arxiv.org/abs/1509.01626

    Number of lines per split:
        - train: 560000
        - test: 38000

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

    :returns: DataPipe that yields tuple of label (1 to 2) and text containing the review
    :rtype: (int, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    def _filepath_fn():
        return os.path.join(root, _PATH)

    def _extracted_filepath_fn():
        return os.path.join(root, _EXTRACTED_FILES[split])

    def _filter_fn(x):
        return _EXTRACTED_FILES[split] in x[0]

    def _modify_res(t):
        return int(t[0]), " ".join(t[1:])

    url_dp = IterableWrapper([URL])

    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=_filepath_fn,
        hash_dict={_filepath_fn(): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=_extracted_filepath_fn)
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b")

    cache_decompressed_dp = cache_decompressed_dp.load_from_tar()

    cache_decompressed_dp = cache_decompressed_dp.filter(_filter_fn)
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return data_dp.parse_csv().map(_modify_res)
