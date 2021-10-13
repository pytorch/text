# Copyright (c) Facebook, Inc. and its affiliates.
import os

from torchdata.datapipes.iter import (
    HttpReader,
    IterableWrapper,
)
from torchtext.data.datasets_utils import (
    _add_docstring_header,
    _create_dataset_directory,
    _wrap_split_argument,
)


NUM_LINES = {
    "train": 67349,
    "dev": 872,
    "test": 1821,
}

MD5 = "9f81648d4199384278b86e315dac217c"
URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

_EXTRACTED_FILES = {
    "train": f"{os.sep}".join(["SST-2", "train.tsv"]),
    "dev": f"{os.sep}".join(["SST-2", "dev.tsv"]),
    "test": f"{os.sep}".join(["SST-2", "test.tsv"]),
}

_EXTRACTED_FILES_MD5 = {
    "train": "da409a0a939379ed32a470bc0f7fe99a",
    "dev": "268856b487b2a31a28c0a93daaff7288",
    "test": "3230e4efec76488b87877a56ae49675a",
}

DATASET_NAME = "SST2"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev", "test"))
def SST2(root, split):
    return SST2Dataset(root, split).get_datapipe()


class SST2Dataset:
    """The SST2 dataset uses torchdata datapipes end-2-end.
    To avoid download at every epoch, we cache the data on-disk
    We do sanity check on dowloaded and extracted data
    """

    def __init__(self, root, split):
        self.root = root
        self.split = split

    def get_datapipe(self):
        # cache data on-disk
        cache_dp = IterableWrapper([URL]).on_disk_cache(
            HttpReader,
            op_map=lambda x: (x[0], x[1].read()),
            filepath_fn=lambda x: os.path.join(self.root, os.path.basename(x)),
        )

        # do sanity check
        check_cache_dp = cache_dp.check_hash(
            {os.path.join(self.root, "SST-2.zip"): MD5}, "md5"
        )

        # extract data from zip
        extracted_files = check_cache_dp.read_from_zip()

        # Filter extracted files and do sanity check
        check_extracted_files = extracted_files.filter(
            lambda x: self.split in x[0]
        ).check_hash(
            {
                os.path.join(
                    self.root, _EXTRACTED_FILES[self.split]
                ): _EXTRACTED_FILES_MD5[self.split]
            },
            "md5",
        )

        # Parse CSV file and yield data samples
        return check_extracted_files.parse_csv(skip_header=True, delimiter="\t").map(
            lambda x: (x[0], x[1])
        )
