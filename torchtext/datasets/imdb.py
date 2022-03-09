import os
from pathlib import Path
from typing import Tuple, Union

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.data.datasets_utils import _wrap_split_argument

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper


URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

MD5 = "7c2ac02c03563afcf9b574c7e56c153a"

NUM_LINES = {
    "train": 25000,
    "test": 25000,
}

_PATH = "aclImdb_v1.tar.gz"

DATASET_NAME = "IMDB"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def IMDB(root: str, split: Union[Tuple[str], str]):
    """IMDB Dataset

    For additional details refer to http://ai.stanford.edu/~amaas/data/sentiment/

    Number of lines per split:
        - train: 25000
        - test: 25000

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

    :returns: DataPipe that yields tuple of label (1 to 2) and text containing the movie review
    :rtype: (int, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])

    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH),
        hash_dict={os.path.join(root, _PATH): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    labels = {"neg", "pos"}
    decompressed_folder = "aclImdb_v1"
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: [os.path.join(root, decompressed_folder, split, label) for label in labels]
    )
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b")
    cache_decompressed_dp = cache_decompressed_dp.read_from_tar()

    def filter_imdb_data(key, fname):
        # eg. fname = "aclImdb/train/neg/12416_3.txt"
        *_, split, label, file = Path(fname).parts
        return key == split and label in labels

    cache_decompressed_dp = cache_decompressed_dp.filter(lambda t: filter_imdb_data(split, t[0]))

    # eg. "aclImdb/train/neg/12416_3.txt" -> "neg"
    cache_decompressed_dp = cache_decompressed_dp.map(lambda t: (Path(t[0]).parts[-2], t[1]))
    cache_decompressed_dp = cache_decompressed_dp.readlines(decode=True)
    cache_decompressed_dp = cache_decompressed_dp.lines_to_paragraphs()  # group by label in cache file
    cache_decompressed_dp = cache_decompressed_dp.map(lambda x: (x[0], x[1].encode()))
    cache_decompressed_dp = cache_decompressed_dp.end_caching(
        mode="wb", filepath_fn=lambda x: os.path.join(root, decompressed_folder, split, x), skip_read=True
    )

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    # get label from cache file, eg. "aclImdb_v1/train/neg" -> "neg"
    return data_dp.readlines().map(lambda t: (Path(t[0]).parts[-1], t[1]))
