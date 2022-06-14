import os
from functools import partial
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper
    from torchtext._download_hooks import GDriveReader

URL = 'https://drive.google.com/u/0/uc?id=0BzQ6rtO2VN95a0c3TlZCWkl3aU0&export=download'

MD5 = "3514e4ab21ab99708ef746581762f71b"

_PATH = "finished_files.zip"

_EXTRACTED_FILES = {
    "train": os.path.join("finished_files", "train.bin"),
    "train": os.path.join("finished_files", "val.bin"),
    "test": os.path.join("finished_files", "test.bin"),
}

_EXTRACTED_FILES_MD5 = {
    "train": "2b5389df76cba2757e2d70627269dbfe",
    "val": "8efa7ac46fc61395d23131ec56c3d9ba",
    "test": "c9b01159cdbb9ff81268c7a3d2278705",
}

DATASET_NAME = "CNNDM"


def _filepath_fn(root: str, _=None):
    return os.path.join(root, _PATH)

def _extracted_filepath_fn(root: str, split: str, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])

def _filter_fn(split: str, x):
    return _EXTRACTED_FILES[split] in x[0]


def CNNDM(root: str, split: Union[Tuple[str], str]):

    url_dp = IterableWrapper([URL])
    
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root): MD5},
        hash_type="md5",
    )

    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_zip().filter(partial(_filter_fn, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    
    return data_dp.routed_decode()

if __name__ == '__main__':

    data = list(CNNDM(os.path.expanduser('~/.torchtext/cache'), 'val'))
    print(type(data))
