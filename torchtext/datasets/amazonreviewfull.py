from torchtext.utils import unicode_csv_reader
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import download_extract_validate
import io
import logging

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'

MD5 = '57d28bd5d930e772930baddf36641c7c'

NUM_LINES = {
    'train': 3000000,
    'test': 650000,
}

_PATH = 'amazon_review_full_csv.tar.gz'

_EXTRACTED_FILES = {
    'train': 'amazon_review_full_csv/train.csv',
    'test': 'amazon_review_full_csv/test.csv'
}

_EXTRACTED_FILES_MD5 = {
    'train': "31b268b09fd794e0ca5a1f59a0358677",
    'test': "0f1e78ab60f625f2a30eab6810ef987c"
}


@add_docstring_header()
@wrap_split_argument(('train', 'test'))
def AmazonReviewFull(root, split):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])

    path = download_extract_validate(root, URL, MD5, _PATH, _EXTRACTED_FILES[split],
                                     _EXTRACTED_FILES_MD5[split], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return RawTextIterableDataset("AmazonReviewFull", NUM_LINES[split],
                                  _create_data_from_csv(path))
