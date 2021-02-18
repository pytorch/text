from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
from torchtext.experimental.datasets.raw.common import find_match
import os
import io
import logging

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'

MD5 = '57d28bd5d930e772930baddf36641c7c'

NUM_LINES = {
    'train': 3000000,
    'test': 650000,
}

_PATH = 'amazon_review_full_csv.tar.gz'


@wrap_split_argument
@add_docstring_header()
def AmazonReviewFull(root='.data', split=('train', 'test'), offset=0):
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
        logging.info('Creating {} data'.format(item))
        datasets.append(RawTextIterableDataset("AmazonReviewFull", NUM_LINES[item],
                                               _create_data_from_csv(path), offset=offset))
    return datasets
