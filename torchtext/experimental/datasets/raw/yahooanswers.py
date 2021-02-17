import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU'

MD5 = 'f3f9899b997a42beb24157e62e3eea8d'

NUM_LINES = {
    'train': 1400000,
    'test': 60000,
}


@wrap_split_argument
@add_docstring_header()
def YahooAnswers(root='.data', split=('train', 'test'), offset=0):
    pass

