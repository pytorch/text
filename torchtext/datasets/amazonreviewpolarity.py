from torchtext.utils import unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _download_extract_validate
import io
import os
import logging

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


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_wrap_split_argument(('train', 'test'))
def AmazonReviewPolarity(root, split):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])
    path = _download_extract_validate(root, URL, MD5, os.path.join(root, _PATH), os.path.join(root, _EXTRACTED_FILES[split]),
                                      _EXTRACTED_FILES_MD5[split], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset("AmazonReviewPolarity", NUM_LINES[split],
                                   _create_data_from_csv(path))
