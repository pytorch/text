import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _find_match
import io

URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'

MD5 = '9ddaacaf6af0710eda8c456decff7832'

NUM_LINES = {
    'train': 1801350,
    'valid': 3760,
    'test': 4358,
}


@_add_docstring_header(num_lines=NUM_LINES)
@_wrap_split_argument(('train', 'valid', 'test'))
def WikiText103(root, split):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    path = _find_match(split, extracted_files)
    logging.info('Creating {} data'.format(split))
    return _RawTextIterableDataset('WikiText103',
                                   NUM_LINES[split], iter(io.open(path, encoding="utf8")))
