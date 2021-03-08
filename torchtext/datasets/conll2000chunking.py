from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _download_extract_validate,
    _create_dataset_directory,
    _create_data_from_iob,
)
import os
import logging

URL = {
    'train': "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz",
    'test': "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz",
}

MD5 = {
    'train': "6969c2903a1f19a83569db643e43dcc8",
    'test': "a916e1c2d83eb3004b38fc6fcd628939",
}

NUM_LINES = {
    'train': 8936,
    'test': 2012,
}

_EXTRACTED_FILES = {
    'train': 'train.txt',
    'test': 'test.txt'
}

_EXTRACTED_FILES_MD5 = {
    'train': "2e2f24e90e20fcb910ab2251b5ed8cd0",
    'test': "56944df34be553b72a2a634e539a0951"
}


DATASET_NAME = "CoNLL2000Chunking"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def CoNLL2000Chunking(root, split):
    # Create a dataset specific subfolder to deal with generic download filenames
    root = os.path.join(root, 'conll2000chunking')
    path = os.path.join(root, split + ".txt.gz")
    data_filename = _download_extract_validate(root, URL[split], MD5[split], path, os.path.join(root, _EXTRACTED_FILES[split]),
                                               _EXTRACTED_FILES_MD5[split], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_iob(data_filename, " "))
