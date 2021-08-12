from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _create_dataset_directory
import os
from pathlib import Path

from datapipes.iter import (
    ReadFilesFromTar,
    HttpReader,
    Saver,
)

from torch.utils.data.datapipes.iter import LoadFilesFromDisk
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
def IMDB(root, split):
    """Demonstrates complex use case where each sample is stored in seperate file and compressed in tar file
        Here we show some fancy filtering and mapping operations.
        Filtering is needed to know which files belong to train/test and neg/pos label
        Mapping is needed to yield proper data samples by extracting label from file name and reading data from file
    """

    # stack saver data pipe on top of web stream to save data to disk
    saver_dp = HttpReader([URL]).map(lambda x: (x[0], x[1].read())).save_to_disk(filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))

    # stack TAR extractor on top of load files data pipe
    extracted_files = LoadFilesFromDisk(saver_dp).read_from_tar()

    # filter the files as applicable to create dataset for given split (train or test)
    filter_files = extracted_files.filter(lambda x: Path(x[0]).parts[-3] == split and Path(x[0]).parts[-2] in ['pos', 'neg'])

    # map the file to yield proper data samples
    return filter_files.map(lambda x: (Path(x[0]).parts[-2], x[1].read().decode('utf-8')))
