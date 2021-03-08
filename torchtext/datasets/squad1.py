from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_json,
)
URL = {
    'train': "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    'dev': "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MD5 = {
    'train': "981b29407e0affa3b1b156f72073b945",
    'dev': "3e85deb501d4e538b6bc56f786231552",
}

NUM_LINES = {
    'train': 87599,
    'dev': 10570,
}


DATASET_NAME = "SQuAD1"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev'))
def SQuAD1(root, split):
    extracted_files = download_from_url(URL[split], root=root, hash_value=MD5[split], hash_type='md5')
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_json(extracted_files))
