import os
from typing import Tuple, Union

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper

URL = "http://mattmahoney.net/dc/enwik9.zip"

MD5 = "3e773f8a1577fda2e27f871ca17f31fd"

_PATH = "enwik9.zip"

NUM_LINES = {"train": 13147026}

DATASET_NAME = "EnWik9"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train",))
def EnWik9(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH),
        hash_dict={os.path.join(root, _PATH): MD5},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    data_dp = FileOpener(cache_dp, mode="b")
    extracted_files = data_dp.read_from_zip()
    return extracted_files.readlines(decode=True, return_path=False)
