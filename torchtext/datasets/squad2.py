from torchtext.utils import download_from_url
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_json,
)
URL = {
    'train': "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    'dev': "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

MD5 = {
    'train': "62108c273c268d70893182d5cf8df740",
    'dev': "246adae8b7002f8679c027697b0b7cf8",
}

NUM_LINES = {
    'train': 130319,
    'dev': 11873,
}


DATASET_NAME = "SQuAD2"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev'))
def SQuAD2(root, split):
    extracted_files = download_from_url(URL[split], root=root, hash_value=MD5[split], hash_type='md5')
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_json(extracted_files))
