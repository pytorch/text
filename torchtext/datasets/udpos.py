import os
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper
    from torchtext._download_hooks import HttpReader

URL = "https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip"

MD5 = "bdcac7c52d934656bae1699541424545"

NUM_LINES = {
    "train": 12543,
    "valid": 2002,
    "test": 2077,
}

_EXTRACTED_FILES = {"train": "train.txt", "valid": "dev.txt", "test": "test.txt"}


DATASET_NAME = "UDPOS"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def UDPOS(root: str, split: Union[Tuple[str], str]):
    """UDPOS Dataset

    Number of lines per split:
        - train: 12543
        - valid: 2002
        - test: 2077

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `valid`, `test`)

    :returns: DataPipe that yields list of words along with corresponding parts-of-speech tags
    :rtype: [list(str), list(str)]
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    def _filepath_fn():
        return os.path.join(root, os.path.basename(URL))

    def _extracted_filepath_fn():
        return os.path.join(root, _EXTRACTED_FILES[split])

    def _filter_fn(x):
        return _EXTRACTED_FILES[split] in x[0]

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=_filepath_fn,
        hash_dict={_filepath_fn(): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=_extracted_filepath_fn)
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").load_from_zip().filter(_filter_fn)
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return data_dp.readlines().read_iob()
