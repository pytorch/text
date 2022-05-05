import os

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _create_dataset_directory,
    _wrap_split_argument,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper

    # we import HttpReader from _download_hooks so we can swap out public URLs
    # with interal URLs when the dataset is used within Facebook
    from torchtext._download_hooks import HttpReader


URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"

MD5 = "4eb0065aba063ef77873d3a9c8088811"

NUM_LINES = {
    "train": 5749,
    "dev": 1500,
    "test": 1379,
}

_PATH = "Stsbenchmark.tar.gz"

DATASET_NAME = "STSB"

_EXTRACTED_FILES = {
    "train": os.path.join("stsbenchmark", "sts-train.csv"),
    "dev": os.path.join("stsbenchmark", "sts-dev.csv"),
    "test": os.path.join("stsbenchmark", "sts-test.csv"),
}


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev", "test"))
def STSB(root, split):
    """STSB Dataset

    For additional details refer to https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark

    Number of lines per split:
        - train: 5749
        - dev: 1500
        - test: 1379

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `dev`, `test`)

    :returns: DataPipe that yields tuple of (index (int), label (float), sentence1 (str), sentence2 (str))
    :rtype: Union[(int, str), (str,)]
    """
    # TODO Remove this after removing conditional dependency
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(x)),
        hash_dict={os.path.join(root, os.path.basename(URL)): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _EXTRACTED_FILES[split])
    )
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").read_from_tar().filter(lambda x: _EXTRACTED_FILES[split] in x[0])
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    parsed_data = data_dp.parse_csv(delimiter='\t').filter(lambda x: len(x) >= 7).map(lambda x: (int(x[3]), float(x[4]), x[5], x[6]))
    return parsed_data
