import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'

MD5 = '9ddaacaf6af0710eda8c456decff7832'

NUM_LINES = {
    'train': 1801350,
    'valid': 3760,
    'test': 4358,
}


@wrap_split_argument
@add_docstring_header()
def WikiText103(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

