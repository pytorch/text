import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import io

URL = 'http://mattmahoney.net/dc/enwik9.zip'

MD5 = '3e773f8a1577fda2e27f871ca17f31fd'

NUM_LINES = {
    'train': 13147026
}


@wrap_split_argument
@add_docstring_header()
def EnWik9(root='.data', split='train', offset=0):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    path = extracted_files[0]
    logging.info('Creating {} data'.format(split[0]))
    return [RawTextIterableDataset('EnWik9',
                                   NUM_LINES[split[0]], iter(io.open(path, encoding="utf8")), offset=offset)]
