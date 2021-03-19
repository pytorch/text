from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _find_match,
    _create_dataset_directory,
    _create_data_from_iob,
)

URL = 'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip'

MD5 = 'bdcac7c52d934656bae1699541424545'

NUM_LINES = {
    'train': 12543,
    'valid': 2002,
    'test': 2077,
}


DATASET_NAME = "UDPOS"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'valid', 'test'))
def UDPOS(root, split):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    if split == 'valid':
        path = _find_match("dev.txt", extracted_files)
    else:
        path = _find_match(split + ".txt", extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_iob(path))
