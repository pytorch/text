import hashlib
import os
from collections import defaultdict
from functools import partial
from typing import Union, Tuple
from benchmark.utils import Timer

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import (
        FileOpener,
        IterableWrapper,
        OnlineReader,
        GDriveReader,
    )

DATASET_NAME = "CNNDM"

URL_LIST = {
    "cnn_train": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_training_urls.txt",
    "cnn_val": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_validation_urls.txt",
    "cnn_test": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_test_urls.txt",
    'dailymail_train': "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_training_urls.txt",
    "dailymail_val": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_validation_urls.txt",
    "dailymail_test": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_test_urls.txt",
}

STORIES_LIST = {
    "cnn": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ",
    "dailymail": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs",
}

PATH_LIST = {
    "cnn": "cnn_stories.tgz",
    "dailymail": "dailymail_stories.tgz",
}

STORIES_MD5 = {"cnn": "85ac23a1926a831e8f46a6b8eaf57263", "dailymail": "f9c5f565e8abe86c38bfa4ae8f96fd72"}

_EXTRACTED_FOLDERS = {
    "cnn": os.path.join("cnn", "stories"),
    "dailymail": os.path.join("dailymail", "stories"),
}

story_fnames = defaultdict(set)


def _filepath_fn(root: str, source: str, _=None):
    return os.path.join(root, PATH_LIST[source])


def _extracted_folder_fn(root: str, source: str, split: str, _=None):
    global story_fnames
    key = source + '_' + split
    story_fnames[key] = set(_get_split_list(source, split))
    filepaths = [os.path.join(root, _EXTRACTED_FOLDERS[source], story) for story in story_fnames[key]]
    return filepaths


def _extracted_filepath_fn(root: str, source: str, x):
    return os.path.join(root, _EXTRACTED_FOLDERS[source], os.path.basename(x))


def _filter_fn(source, split, x):
    return os.path.basename(x[0]) in story_fnames[source + '_' + split]


def _hash_urls(s):
    """
    Returns story filename as a heximal formated SHA1 hash of the input url string.
    Code is inspired from https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
    """
    url = s[1]
    h = hashlib.sha1()
    h.update(url)
    url_hash = h.hexdigest()
    story_fname = url_hash + ".story"
    return story_fname


def _get_split_list(source:str, split: str):
    url_dp = IterableWrapper([URL_LIST[source + '_' + split]])
    online_dp = OnlineReader(url_dp)
    return online_dp.readlines().map(fn=_hash_urls)


def _load_stories(root: str, source: str, split: str):
    story_dp = IterableWrapper([STORIES_LIST[source]])
    cache_compressed_dp = story_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root, source),
        hash_dict={_filepath_fn(root, source): STORIES_MD5[source]},
        hash_type="md5",
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_folder_fn, root, source, split))
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_tar().filter(partial(_filter_fn, source, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", filepath_fn=partial(_extracted_filepath_fn, root, source))
    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return cache_decompressed_dp


#@_create_dataset_directory(dataset_name=DATASET_NAME)
#@_wrap_split_argument(("train", "val", "test"))
def CNNDM(root: str, split: Union[Tuple[str], str]):
    """CNNDM Dataset

    .. warning::

        Using datapipes is still currently subject to a few caveats. If you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://arxiv.org/pdf/1704.04368.pdf

    Number of lines per split:
        - train: 287,227
        - val: 13,368
        - test: 11,490

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `val`, `test`)

    :returns: DataPipe that yields a tuple of texts containing an article and its abstract (i.e. (article, abstract))
    :rtype: (str, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    cnn_dp = _load_stories(root, "cnn", split)
    dailymail_dp = _load_stories(root, "dailymail", split)
    data_dp = cnn_dp.concat(dailymail_dp)
    return data_dp.parse_cnndm_data().shuffle().set_shuffle(False).sharding_filter()

if __name__ == '__main__':

    out = CNNDM(os.path.expanduser('~/.torchtext/cache'), 'train')
    with Timer(f"initialize dataset"):
        ex = iter(out)
    
    for i in range(2):
        with Timer(f"iteration: {i}"):
            next(ex)