from typing import Tuple, Union

from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper

import os

from torchtext.data.datasets_utils import _add_docstring_header, _create_dataset_directory, _wrap_split_argument

URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k"

MD5 = "dca7b1ae12b1091090db52aa7ec5ca64"

NUM_LINES = {
    "train": 560000,
    "test": 70000,
}

_PATH = "dbpedia_csv.tar.gz"

_EXTRACTED_FILES = {"train": os.path.join("dbpedia_csv", "train.csv"), "test": os.path.join("dbpedia_csv", "test.csv")}

DATASET_NAME = "DBpedia"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=14)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def DBpedia(root: str, split: Union[Tuple[str], str]):
    # TODO Remove this after removing conditional dependency
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])

    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH), hash_dict={os.path.join(root, _PATH): MD5}, hash_type="md5"
    )
    cache_dp = GDriveReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, mode="b")

    extracted_files = cache_dp.read_from_tar()

    filter_extracted_files = extracted_files.filter(lambda x: _EXTRACTED_FILES[split] in x[0])

    return filter_extracted_files.parse_csv().map(fn=lambda t: (int(t[0]), " ".join(t[1:])))
