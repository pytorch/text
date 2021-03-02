from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _download_extract_validate
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


def _create_data_from_iob(data_path, separator):
    with open(data_path, encoding="utf-8") as input_file:
        columns = []
        for line in input_file:
            line = line.strip()
            if line == "":
                if columns:
                    yield columns
                columns = []
            else:
                for i, column in enumerate(line.split(separator)):
                    if len(columns) < i + 1:
                        columns.append([])
                    columns[i].append(column)
        if len(columns) > 0:
            yield columns


@_add_docstring_header(num_lines=NUM_LINES)
@_wrap_split_argument(('train', 'test'))
def CoNLL2000Chunking(root, split):
    # Create a dataset specific subfolder to deal with generic download filenames
    root = os.path.join(root, 'conll2000chunking')
    path = os.path.join(root, split + ".txt.gz")
    data_filename = _download_extract_validate(root, URL[split], MD5[split], path, os.path.join(root, _EXTRACTED_FILES[split]),
                                               _EXTRACTED_FILES_MD5[split], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset("CoNLL2000Chunking", NUM_LINES[split],
                                   _create_data_from_iob(data_filename, " "))
