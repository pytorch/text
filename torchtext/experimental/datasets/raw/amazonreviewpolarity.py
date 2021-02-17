import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM'

MD5 = 'fe39f8b653cada45afd5792e0f0e8f9b'

NUM_LINES = {
    'train': 3600000,
    'test': 400000,
}


@wrap_split_argument
@add_docstring_header()
def AmazonReviewPolarity(root='.data', split=('train', 'test'), offset=0):
    pass

