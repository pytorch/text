from torchtext._internal.module_utils import is_module_available
from typing import Union, Tuple
if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper

from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM'

MD5 = 'fe39f8b653cada45afd5792e0f0e8f9b'

NUM_LINES = {
    'train': 3600000,
    'test': 400000,
}

_PATH = 'amazon_review_polarity_csv.tar.gz'

_EXTRACTED_FILES = {
    'train': os.path.join('amazon_review_polarity_csv', 'train.csv'),
    'test': os.path.join('amazon_review_polarity_csv', 'test.csv'),
}


DATASET_NAME = "AmazonReviewPolarity"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AmazonReviewPolarity(root: str, split: Union[Tuple[str], str]):
    # TODO Remove this after removing conditional dependency
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH), hash_dict={os.path.join(root, _PATH): MD5}, hash_type="md5"
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    def extracted_filepath_fn(x):
        file_path = os.path.join(root, _EXTRACTED_FILES[split])
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return file_path

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=extracted_filepath_fn)
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").\
        read_from_tar().\
        filter(lambda x: _EXTRACTED_FILES[split] in x[0]).\
        map(lambda x: (x[0].replace('_PATH' + '/', ''), x[1]))
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    data_dp = FileOpener(cache_decompressed_dp, mode='b')

    return data_dp.parse_csv().map(fn=lambda t: (int(t[0]), ' '.join(t[1:])))
