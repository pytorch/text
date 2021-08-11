from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _create_dataset_directory
import os
from pathlib import Path

from datapipes.iter import (
    IterableAsDataPipe,
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
    saver_dp = HttpReader([URL]).map(lambda x: (x[0],x[1].read())).save_to_disk(filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))
    extracted_files = LoadFilesFromDisk(saver_dp).read_from_tar()
    return extracted_files.filter(lambda x: Path(x[0]).parts[-3] == split and Path(x[0]).parts[-2]
                                  in ['pos', 'neg']).map(lambda x: (Path(x[0]).parts[-2], x[1].read().decode('utf-8')))
