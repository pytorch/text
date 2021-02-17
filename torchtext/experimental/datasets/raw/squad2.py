import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = {
    'train': "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    'dev': "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

MD5 = {
    'train': "62108c273c268d70893182d5cf8df740",
    'dev': "246adae8b7002f8679c027697b0b7cf8",
}

NUM_LINES = {
    'train': 130319,
    'dev': 11873,
}


@wrap_split_argument
@add_docstring_header()
def SQuAD2(root='.data', split=('train', 'dev'), offset=0):
    pass

