import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0'

MD5 = 'f7ddfafed1033f68ec72b9267863af6c'

NUM_LINES = {
    'train': 650000,
    'test': 50000,
}


@wrap_split_argument
@add_docstring_header()
def YelpReviewFull(root='.data', split=('train', 'test'), offset=0):
    pass

