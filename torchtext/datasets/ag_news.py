from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)

from datapipes.iter import (
    CSVParser,
    HttpReader
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

DATASET_NAME = "AG_NEWS"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=4)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def AG_NEWS(root, split):
    # TODO Caching mechanism
    return HttpReader([URL[split]]).parse_csv_files().map(lambda t: (int(t[1]), ' '.join(t[2:])))
