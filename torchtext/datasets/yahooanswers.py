from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import find_match
import os
import io

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU'

MD5 = 'f3f9899b997a42beb24157e62e3eea8d'

NUM_LINES = {
    'train': 1400000,
    'test': 60000,
}

_PATH = 'yahoo_answers_csv.tar.gz'


@wrap_split_argument
@add_docstring_header()
def YahooAnswers(root='.data', split=('train', 'test'), offset=0):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    datasets = []
    for item in split:
        path = find_match(item + '.csv', extracted_files)
        datasets.append(RawTextIterableDataset("YahooAnswers", NUM_LINES[item],
                                               _create_data_from_csv(path), offset=offset))
    return datasets
