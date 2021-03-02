import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
import io

URL = 'http://mattmahoney.net/dc/enwik9.zip'

MD5 = '3e773f8a1577fda2e27f871ca17f31fd'

NUM_LINES = {
    'train': 13147026
}


@_add_docstring_header(num_lines=NUM_LINES)
@_wrap_split_argument(('train',))
def EnWik9(root, split):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    path = extracted_files[0]
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset('EnWik9',
                                   NUM_LINES[split], iter(io.open(path, encoding="utf8")))
