from torchtext.utils import unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _download_extract_validate
import io
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


@_add_docstring_header(num_lines=NUM_LINES, num_classes=5)
@_wrap_split_argument(('train', 'test'))
def SogouNews(root, split):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])
    path = _download_extract_validate(root, URL, MD5, os.path.join(root, _PATH), os.path.join(root, _EXTRACTED_FILES[split]),
                                      _EXTRACTED_FILES_MD5[split], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset("SogouNews", NUM_LINES[split],
                                   _create_data_from_csv(path))
