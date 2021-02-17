import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = {
    'train': "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    'test': "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    'valid': "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
}

MD5 = {
    'train': "f26c4b92c5fdc7b3f8c7cdcb991d8420",
    'valid': "aa0affc06ff7c36e977d7cd49e3839bf",
    'test': "8b80168b89c18661a38ef683c0dc3721",
}

NUM_LINES = {
    'train': 42068,
    'valid': 3370,
    'test': 3761,
}


@wrap_split_argument
@add_docstring_header()
def PennTreebank(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

