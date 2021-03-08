from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _find_match
from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.data.datasets_utils import _create_data_from_csv
import os

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU'

MD5 = 'f3f9899b997a42beb24157e62e3eea8d'

NUM_LINES = {
    'train': 1400000,
    'test': 60000,
}

_PATH = 'yahoo_answers_csv.tar.gz'

DATASET_NAME = "YahooAnswers"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=10)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def YahooAnswers(root, split):
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    path = _find_match(split + '.csv', extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_csv(path))
