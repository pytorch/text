from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _download_extract_validate,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os
import logging

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE'

MD5 = '0c1700ba70b73f964dd8de569d3fd03e'

NUM_LINES = {
    'train': 450000,
    'test': 60000,
}

_PATH = 'sogou_news_csv.tar.gz'

_EXTRACTED_FILES = {
    'train': f'{os.sep}'.join(['sogou_news_csv', 'train.csv']),
    'test': f'{os.sep}'.join(['sogou_news_csv', 'test.csv']),
}

_EXTRACTED_FILES_MD5 = {
    'train': "f36156164e6eac2feda0e30ad857eef0",
    'test': "59e493c41cee050329446d8c45615b38"
}

DATASET_NAME = "SogouNews"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=5)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def SogouNews(root, split):
    path = _download_extract_validate(root, URL, MD5, os.path.join(root, _PATH), os.path.join(root, _EXTRACTED_FILES[split]),
                                      _EXTRACTED_FILES_MD5[split], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_csv(path))
