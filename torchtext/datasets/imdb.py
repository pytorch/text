from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _create_dataset_directory
import os
from pathlib import Path

from datapipes.iter import ReadFilesFromTar
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
    dataset_tar = download_from_url(URL, root=root,
                                    hash_value=MD5, hash_type='md5')

    extracted_files = ReadFilesFromTar([(os.path.dirname(dataset_tar), open(dataset_tar, 'rb'))])
    return extracted_files.filter(lambda x: Path(x[0]).parts[-3] == split and Path(x[0]).parts[-2]
                                  in ['pos', 'neg']).map(lambda x: (Path(x[0]).parts[-2], x[1].read().decode('utf-8')))
