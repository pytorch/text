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
from torchtext.data.data_pipes import GDriveReaderDataPipe as GDriveReader

from torch.utils.data.datapipes.iter import LoadFilesFromDisk
URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'

MD5 = '57d28bd5d930e772930baddf36641c7c'

NUM_LINES = {
    'train': 3000000,
    'test': 650000,
}

_PATH = 'amazon_review_full_csv.tar.gz'

_EXTRACTED_FILES = {
    'train': f'{os.sep}'.join(['amazon_review_full_csv', 'train.csv']),
    'test': f'{os.sep}'.join(['amazon_review_full_csv', 'test.csv']),
}

_EXTRACTED_FILES_MD5 = {
    'train': "31b268b09fd794e0ca5a1f59a0358677",
    'test': "0f1e78ab60f625f2a30eab6810ef987c"
}

DATASET_NAME = "AmazonReviewFull"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=5)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def AmazonReviewFull(root, split):
    saver_dp = GDriveReader([URL]).map(lambda x: (x[0],x[1].read())).save_to_disk(filepath_fn=lambda x: os.path.join(root, x))
    extracted_files = LoadFilesFromDisk(saver_dp).read_from_tar()
    return extracted_files.filter(lambda x: split in x[0]).parse_csv_files().map(lambda t: (int(t[1]), ' '.join(t[2:])))
