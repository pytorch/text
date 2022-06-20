import os
from functools import partial
from typing import Union, Tuple

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
    "train": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt",
    "val": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt",
    "test": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt",
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
    "daily_mail": os.path.join("dailymail", "stories"),
}


def _filepath_fn(root: str, source: str, _=None):
    return os.path.join(root, PATH_LIST[source])


# this function will be used to cache the contents of the tar file
def _extracted_filepath_fn(root: str, source: str, t):
    return os.path.join(root, _EXTRACTED_FOLDERS[source])


def _filter_fn(story_fnames, x):
    return os.path.basename(x[0]) in story_fnames


def _get_split_list(split: str):
    url_dp = IterableWrapper([URL_LIST[split]])
    online_dp = OnlineReader(url_dp)
    return online_dp.readlines().parse_cnndm_split()


def _load_stories(root: str, source: str):
    story_dp = IterableWrapper([STORIES_LIST[source]])
    cache_compressed_dp = story_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root, source),
        hash_dict={_filepath_fn(root, source): STORIES_MD5[source]},
        hash_type="md5",
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)
    # TODO: cache the contents of the extracted tar file
    cache_decompressed_dp = FileOpener(cache_compressed_dp, mode="b").load_from_tar()
    return cache_decompressed_dp


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "val", "test"))
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

    cnn_dp = _load_stories(root, "cnn")
    dailymail_dp = _load_stories(root, "dailymail")
    data_dp = cnn_dp.concat(dailymail_dp)
    # TODO: store the .story filenames corresponding to each split on disk so we can pass that into the filepath_fn
    # of the on_disk_cache_dp which caches the files extracted from the tar
    story_fnames = set(_get_split_list(split))
    data_dp = data_dp.filter(partial(_filter_fn, story_fnames))
    return data_dp.parse_cnndm_data().shuffle().set_shuffle(False).sharding_filter()
