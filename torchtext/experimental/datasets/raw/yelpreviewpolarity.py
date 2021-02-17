import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg'

MD5 = '620c8ae4bd5a150b730f1ba9a7c6a4d3'

NUM_LINES = {
    'train': 560000,
    'test': 38000,
}


@wrap_split_argument
@add_docstring_header()
def YelpReviewPolarity(root='.data', split=('train', 'test'), offset=0):
    pass

