import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'

MD5 = '64150a352f3abe890a87f6c6838524a6'

NUM_LINES = {
    'train': 17676013,
}


@wrap_split_argument
@add_docstring_header()
def WMTNewsCrawl(root='.data', split=train, offset=0):
    pass

