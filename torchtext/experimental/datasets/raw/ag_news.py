import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
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


@wrap_split_argument
@add_docstring_header()
def AG_NEWS(root='.data', split=('train', 'test'), offset=0):
    extracted_files = [download_from_url(URL[item], root=root,
                                         path=os.path.join(root, _PATHS[dataset_name][item]),
                                         hash_value=MD5['AG_NEWS'][item],
                                         hash_type='md5') for item in ('train', 'test')]
    datasets = []
    for item in split:
        path = find_match(item + '.csv', extracted_files)
        datasets.append(RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name][item],
                                               _create_data_from_csv(path), offset=offset))
    return datasets

