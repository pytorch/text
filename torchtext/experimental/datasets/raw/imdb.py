from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import io

URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

NUM_LINES = {
    'train': 25000,
    'test': 25000,
}

_PATH = 'aclImdb_v1.tar.gz'


@wrap_split_argument
@add_docstring_header()
def IMDB(root='.data', split=('train', 'test'), offset=0):
    def generate_imdb_data(key, extracted_files):
        for fname in extracted_files:
            if 'urls' in fname:
                continue
            elif key in fname and ('pos' in fname or 'neg' in fname):
                with io.open(fname, encoding="utf8") as f:
                    label = 'pos' if 'pos' in fname else 'neg'
                    yield label, f.read()
    dataset_tar = download_from_url(URL, root=root,
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    datasets = []
    for item in split:
        iterator = generate_imdb_data(item, extracted_files)
        datasets.append(RawTextIterableDataset("IMDB", NUM_LINES[item], iterator, offset=offset))
    return datasets
