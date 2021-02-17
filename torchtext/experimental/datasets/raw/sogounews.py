import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE'

MD5 = '0c1700ba70b73f964dd8de569d3fd03e'

NUM_LINES = {
    'train': 450000,
    'test': 60000,
}


@wrap_split_argument
@add_docstring_header()
def SogouNews(root='.data', split=('train', 'test'), offset=0):
    pass

