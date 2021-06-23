from torchtext.utils import (
    download_from_url,
)
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os

URL = {
    'train': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    'test': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

MD5 = {
    'train': "b1a00f826fdfbd249f79597b59e1dc12",
    'test': "d52ea96a97a2d943681189a97654912d",
}

NUM_LINES = {
    'train': 120000,
    'test': 7600,
}

OFFSETS = {
    'train': {
        0: 0,
        10000: 2513230,
        20000: 4989675,
        30000: 7448280,
        40000: 9917610,
        50000: 12373034,
        60000: 14841577,
        70000: 17310009,
        80000: 19755260,
        90000: 22180771,
        100000: 24603053,
        110000: 27027049,
    },
    'test': {0: 0},
}

DATASET_NAME = "AG_NEWS"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=4)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def AG_NEWS(root, split, offset=0):
    path = download_from_url(URL[split], root=root,
                             path=os.path.join(root, split + ".csv"),
                             hash_value=MD5[split],
                             hash_type='md5')
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_csv(path, offset, OFFSETS[split]))
