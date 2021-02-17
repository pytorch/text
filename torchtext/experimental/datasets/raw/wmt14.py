import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'

MD5 = '874ab6bbfe9c21ec987ed1b9347f95ec'

NUM_LINES = {
    'train': 4500966,
    'valid': 3000,
    'test': 3003,
}


@wrap_split_argument
@add_docstring_header()
def WMT14(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

