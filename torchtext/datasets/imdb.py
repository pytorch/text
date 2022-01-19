import os
from pathlib import Path
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.data.datasets_utils import _wrap_split_argument

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper, HttpReader


URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

NUM_LINES = {
    'train': 25000,
    'test': 25000,
}

_PATH = 'aclImdb_v1.tar.gz'

DATASET_NAME = "IMDB"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def IMDB(root: str, split: Union[Tuple[str], str]):
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError("Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`")

    url_dp = IterableWrapper([URL])

    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH),
        hash_dict={os.path.join(root, _PATH): MD5}, hash_type="md5"
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, mode="b")

    extracted_files = cache_dp.read_from_tar()

    def filter_imdb_data(key, fname):
        *_, split, label, file = Path(fname).parts
        return key == split and (label in ['pos', 'neg'])

    filter_extracted_files = extracted_files.filter(lambda t: filter_imdb_data(split, t[0]))

    return filter_extracted_files.readlines(decode=True).map(lambda t: (Path(t[0]).parts[-2], t[1]))
