from torchtext._internal.module_utils import is_module_available

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper

import os
import functools
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)
from typing import Union, Tuple
from pathlib import Path

URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'

MD5 = '9ddaacaf6af0710eda8c456decff7832'

NUM_LINES = {
    'train': 1801350,
    'valid': 3760,
    'test': 4358,
}

DATASET_NAME = "WikiText103"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'valid', 'test'))
def WikiText103(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")
    url_dp = IterableWrapper([URL])
    # cache data on-disk
    filepath_fn = functools.partial(lambda x: os.path.join(root, os.path.basename(x)))
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=filepath_fn,
        hash_dict={os.path.join(root, os.path.basename(URL)): MD5},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, mode="b")
    # stack Zip extractor on top of load files data pipe
    extracted_files = cache_dp.read_from_zip()
    # filter the files as applicable to create dataset for given split (train or test)
    filter_fn = functools.partial(lambda x: split in Path(x[0]).parts[-1])
    filter_extracted_files = extracted_files.filter(filter_fn)
    extract_text_fn = functools.partial(lambda t: t[1].decode())
    return filter_extracted_files.readlines(strip_newline=False).map(extract_text_fn)
