import os

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _create_dataset_directory

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper


URL = "http://mattmahoney.net/dc/enwik9.zip"

MD5 = "3e773f8a1577fda2e27f871ca17f31fd"

_PATH = "enwik9.zip"

NUM_LINES = {"train": 13147026}

DATASET_NAME = "EnWik9"


@_create_dataset_directory(dataset_name=DATASET_NAME)
def EnWik9(root: str):
    """EnWik9 dataset

    For additional details refer to http://mattmahoney.net/dc/textdata.html

    Number of lines in dataset: 13147026

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')

    :returns: DataPipe that yields raw text rows from WnWik9 dataset
    :rtype: str
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

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.splitext(_PATH)[0])
    )
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").read_from_zip()
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return data_dp.readlines(return_path=False)
