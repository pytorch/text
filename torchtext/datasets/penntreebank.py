import os
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper


URL = {
    "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    "valid": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
}

MD5 = {
    "train": "f26c4b92c5fdc7b3f8c7cdcb991d8420",
    "valid": "aa0affc06ff7c36e977d7cd49e3839bf",
    "test": "8b80168b89c18661a38ef683c0dc3721",
}

NUM_LINES = {
    "train": 42068,
    "valid": 3370,
    "test": 3761,
}

DATASET_NAME = "PennTreebank"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def PennTreebank(root, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL[split]])
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(x)),
        hash_dict={os.path.join(root, os.path.basename(URL[split])): MD5[split]},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    data_dp = FileOpener(cache_dp, mode="r")
    # remove single leading and trailing space from the dataset
    return data_dp.readlines(return_path=False).map(lambda t: t.strip())
