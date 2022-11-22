import os
from functools import partial

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import HttpReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _create_dataset_directory

URL = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"

MD5 = "b6d5672bd9dc1e66ab2bb020ebeafb8d"

_PATH = "quora_duplicate_questions.tsv"

NUM_LINES = {"train": 404290}

DATASET_NAME = "QQP"


def _filepath_fn(root, _=None):
    return os.path.join(root, _PATH)


def _modify_res(x):
    return (int(x[-1]), x[3], x[4])


@_create_dataset_directory(dataset_name=DATASET_NAME)
def QQP(root: str):
    """QQP dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')

    :returns: DataPipe that yields rows from QQP dataset (label (int), question1 (str), question2 (str))
    :rtype: (int, str, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL])
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root): MD5},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_dp = FileOpener(cache_dp, encoding="utf-8")
    # some context stored at top of the file needs to be removed
    parsed_data = cache_dp.parse_csv(skip_lines=1, delimiter="\t").map(_modify_res)
    return parsed_data.shuffle().set_shuffle(False).sharding_filter()
