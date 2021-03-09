import logging
from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _read_text_iterator,
)

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

DATASET_NAME = "PennTreebank"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'valid', 'test'))
def PennTreebank(root, split):
    path = download_from_url(URL[split],
                             root=root, hash_value=MD5[split],
                             hash_type='md5')
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset(DATASET_NAME,
                                   NUM_LINES[split],
                                   _read_text_iterator(path))
