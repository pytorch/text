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

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k'

MD5 = 'dca7b1ae12b1091090db52aa7ec5ca64'

NUM_LINES = {
    'train': 560000,
    'test': 70000,
}

_PATH = 'dbpedia_csv.tar.gz'

DATASET_NAME = "DBpedia"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=14)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def DBpedia(root, split):
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    path = _find_match(split + '.csv', extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_csv(path))
