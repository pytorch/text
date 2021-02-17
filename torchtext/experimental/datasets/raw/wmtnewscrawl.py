import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
from torchtext.experimental.datasets.raw.common import find_match
import io

URL = 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'

MD5 = '64150a352f3abe890a87f6c6838524a6'

NUM_LINES = {
    'train': 17676013,
}


@wrap_split_argument
@add_docstring_header()
def WMTNewsCrawl(root='.data', split=train, offset=0):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    file_name = 'news.{}.{}.shuffled'.format(year, language)
    extracted_files = [f for f in extracted_files if file_name in f]


    datasets = []
    for item in split:
        path = find_match(item, extracted_files)
        logging.info('Creating {} data'.format(item))
        datasets.append(RawTextIterableDataset('WMTNewsCrawl',
                                               NUM_LINES[item], iter(io.open(path, encoding="utf8")), offset=offset))
