import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k'

MD5 = 'dca7b1ae12b1091090db52aa7ec5ca64'

NUM_LINES = {
    'train': 560000,
    'test': 70000,
}


@wrap_split_argument
@add_docstring_header()
def DBpedia(root='.data', split=('train', 'test'), offset=0):
    pass

