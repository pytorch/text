from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import find_match
import os
import io

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg'

MD5 = '620c8ae4bd5a150b730f1ba9a7c6a4d3'

NUM_LINES = {
    'train': 560000,
    'test': 38000,
}

_PATH = 'yelp_review_polarity_csv.tar.gz'


@add_docstring_header(num_lines=NUM_LINES)
@wrap_split_argument(('train', 'test'))
def YelpReviewPolarity(root, split):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    path = find_match(split + '.csv', extracted_files)
    return RawTextIterableDataset("YelpReviewPolarity", NUM_LINES[split],
                                  _create_data_from_csv(path))
