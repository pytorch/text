import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = 'https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8'

MD5 = 'c393ed3fc2a1b0f004b3331043f615ae'

NUM_LINES = {
    'train': 196884,
    'valid': 993,
    'test': 1305,
}


@wrap_split_argument
@add_docstring_header()
def IWSLT(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

