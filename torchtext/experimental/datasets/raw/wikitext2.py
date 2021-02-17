import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

MD5 = '542ccefacc6c27f945fb54453812b3cd'

NUM_LINES = {
    'train': 36718,
    'valid': 3760,
    'test': 4358,
}


@wrap_split_argument
@add_docstring_header()
def WikiText2(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

