import os
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper


URL = {
    "train": r"http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz",
    "valid": r"http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz",
    "test": r"http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz",
}

MD5 = {
    "train": "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e",
    "valid": "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c",
    "test": "0681be16a532912288a91ddd573594fbdd57c0fbb81486eff7c55247e35326c2",
}

_PREFIX = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

NUM_LINES = {
    "train": 29000,
    "valid": 1014,
    "test": 1000,
}

DATASET_NAME = "Multi30k"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def Multi30k(root: str, split: Union[Tuple[str], str], language_pair: Tuple[str] = ("de", "en")):
    """Multi30k dataset

    For additional details refer to https://www.statmt.org/wmt16/multimodal-task.html#task1

    Number of lines per split:
        - train: 29000
        - valid: 1014
        - test: 1000

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: ('train', 'valid', 'test')
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')

    :return: DataPipe that yields tuple of source and target sentences
    :rtype: (str, str)
    """

    assert len(language_pair) == 2, "language_pair must contain only 2 elements: src and tgt language respectively"
    assert tuple(sorted(language_pair)) == (
        "de",
        "en",
    ), "language_pair must be either ('de','en') or ('en', 'de')"

    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at `https://github.com/pytorch/data`"
        )

    url_dp = IterableWrapper([URL[split]])

    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(URL[split])),
        hash_dict={os.path.join(root, os.path.basename(URL[split])): MD5[split]},
        hash_type="sha256",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)

    src_cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, f"{_PREFIX[split]}.{language_pair[0]}")
    )
    src_cache_decompressed_dp = (
        FileOpener(src_cache_decompressed_dp, mode="b")
        .read_from_tar()
        .filter(lambda x: f"{_PREFIX[split]}.{language_pair[0]}" in x[0])
    )
    src_cache_decompressed_dp = src_cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    tgt_cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, f"{_PREFIX[split]}.{language_pair[1]}")
    )
    tgt_cache_decompressed_dp = (
        FileOpener(tgt_cache_decompressed_dp, mode="b")
        .read_from_tar()
        .filter(lambda x: f"{_PREFIX[split]}.{language_pair[1]}" in x[0])
    )
    tgt_cache_decompressed_dp = tgt_cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    src_data_dp = FileOpener(src_cache_decompressed_dp, encoding="utf-8").readlines(
        return_path=False, strip_newline=True
    )
    tgt_data_dp = FileOpener(tgt_cache_decompressed_dp, encoding="utf-8").readlines(
        return_path=False, strip_newline=True
    )

    return src_data_dp.zip(tgt_data_dp)
