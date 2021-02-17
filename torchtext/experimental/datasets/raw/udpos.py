import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip'

MD5 = 'bdcac7c52d934656bae1699541424545'

NUM_LINES = {
    'train': 12543,
    'valid': 2002,
    'test': 2077,
}


@wrap_split_argument
@add_docstring_header()
def UDPOS(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

