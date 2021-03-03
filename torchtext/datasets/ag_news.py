from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
import os
import io

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


@_add_docstring_header(num_lines=NUM_LINES, num_classes=4)
@_wrap_split_argument(('train', 'test'))
def AG_NEWS(root, split):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])

    path = download_from_url(URL[split], root=root,
                             path=os.path.join(root, split + ".csv"),
                             hash_value=MD5[split],
                             hash_type='md5')
    return _RawTextIterableDataset("AG_NEWS", NUM_LINES[split],
                                   _create_data_from_csv(path))
