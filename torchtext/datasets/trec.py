from torchtext.utils import download_from_url
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header
import os

URL = {
    'train': 'http://cogcomp.org/Data/QA/QC/train_5500.label',
    'test': 'http://cogcomp.org/Data/QA/QC/TREC_10.label',
}

MD5 = {
    'train': '073462e3fcefaae31e00edb1f18d2d02',
    'test': '323a3554401d86e650717e2d2f942589',
}

NUM_LINES = {
    'train': 5452,
    'test': 500,
}


@wrap_split_argument
@add_docstring_header()
def Trec(root='.data', split=('train', 'test'), offset=0):
    def _create_data_from_file(data_path):
        for line in open(os.path.expanduser(data_path), 'rb'):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b'\xf0', b' ').decode().partition(' ')
            label = label.split(":")[1]
            yield label, text

    datasets = []
    for item in split:
        data_path = download_from_url(URL[item], root=root,
                                      hash_value=MD5[item], hash_type='md5')
        datasets.append(RawTextIterableDataset("Trec", NUM_LINES[item],
                                               _create_data_from_file(data_path), offset=offset))
    return datasets
