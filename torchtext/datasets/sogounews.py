from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header
from torchtext.datasets.common import find_match
import os
import io

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE'

MD5 = '0c1700ba70b73f964dd8de569d3fd03e'

NUM_LINES = {
    'train': 450000,
    'test': 60000,
}

_PATH = 'sogou_news_csv.tar.gz'


@wrap_split_argument
@add_docstring_header()
def SogouNews(root='.data', split=('train', 'test'), offset=0):
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
        datasets.append(RawTextIterableDataset("SogouNews", NUM_LINES[item],
                                               _create_data_from_csv(path), offset=offset))
    return datasets
