import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = {
    'train': "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    'dev': "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MD5 = {
    'train': "981b29407e0affa3b1b156f72073b945",
    'dev': "3e85deb501d4e538b6bc56f786231552",
}

NUM_LINES = {
    'train': 87599,
    'dev': 10570,
}


@wrap_split_argument
@add_docstring_header()
def SQuAD1(root='.data', split=('train', 'dev'), offset=0):
    pass

