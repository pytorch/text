from torchtext.utils import (
    download_from_url,
    extract_archive,
)
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _find_match,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0'

MD5 = 'f7ddfafed1033f68ec72b9267863af6c'

NUM_LINES = {
    'train': 650000,
    'test': 50000,
}

_PATH = 'yelp_review_full_csv.tar.gz'

DATASET_NAME = "YelpReviewFull"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=5)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def YelpReviewFull(root, split):
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    path = _find_match(split + '.csv', extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_csv(path))
