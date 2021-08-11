from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)
import os
import logging
from datapipes.iter import(
    CSVParser,
    ReadFilesFromTar,
    HttpReader,
)

from torchtext.data.data_pipes import GDriveReaderDataPipe as GDriveReader

from torch.utils.data.datapipes.iter import LoadFilesFromDisk


URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM'

MD5 = 'fe39f8b653cada45afd5792e0f0e8f9b'

NUM_LINES = {
    'train': 3600000,
    'test': 400000,
}

_PATH = 'amazon_review_polarity_csv.tar.gz'

_EXTRACTED_FILES = {
    'train': f'{os.sep}'.join(['amazon_review_polarity_csv', 'train.csv']),
    'test': f'{os.sep}'.join(['amazon_review_polarity_csv', 'test.csv']),
}

_EXTRACTED_FILES_MD5 = {
    'train': "520937107c39a2d1d1f66cd410e9ed9e",
    'test': "f4c8bded2ecbde5f996b675db6228f16"
}

DATASET_NAME = "AmazonReviewPolarity"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def AmazonReviewPolarity(root, split):
    saver_dp = GDriveReader([URL]).map(lambda x: (x[0],x[1].read())).save_to_disk(filepath_fn=lambda x: os.path.join(root, x))
    extracted_files = LoadFilesFromDisk(saver_dp).read_from_tar()
    return extracted_files.filter(lambda x: split in x[0]).parse_csv_files().map(lambda t: (int(t[1]), ' '.join(t[2:])))

