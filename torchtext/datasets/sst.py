from torchtext.utils import download_from_url, extract_archive
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header
from torchtext.datasets.common import find_match
import os
import io
try:
    from nltk.tree import Tree
except ImportError:
    print("Please install NLTK. See the docs at https://nltk.org for more information.")


URL = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'

MD5 = '5c3b23ce88089053fd3efeb4a34f614d'

NUM_LINES = {
    'train': 8544,
    'dev': 1101,
    'test': 2210,
}

_PATH = 'trainDevTestTrees_PTB.zip'


@wrap_split_argument
@add_docstring_header()
def SST(root='.data', split=('train', 'dev', 'test'), offset=0):
    def _create_data_from_tree(data_path):
        with io.open(data_path, encoding="utf8") as f:
            for line in f:
                tree = Tree.fromstring(line)
                yield tree.label(), ' '.join(tree.leaves())
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    datasets = []
    for item in split:
        path = find_match(item + '.txt', extracted_files)
        datasets.append(RawTextIterableDataset("SST", NUM_LINES[item],
                                               _create_data_from_tree(path), offset=offset))
    return datasets
