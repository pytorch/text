import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import construct_dataset_description
import io

URL = 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'

MD5 = 'c70da2ba79db33fb0fc9119cbad16260'

NUM_LINES = {
    'train': 17676013,
}

_AVAILABLE_YEARS = [2010]
_AVAILABLE_LANGUAGES = [
    "cs",
    "en",
    "fr",
    "es",
    "de"
]


@add_docstring_header()
@wrap_split_argument(('train',))
def WMTNewsCrawl(root, split, year=2010, language='en'):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    if year not in _AVAILABLE_YEARS:
        raise ValueError("{} not available. Please choose from years {}".format(year, _AVAILABLE_YEARS))
    if language not in _AVAILABLE_LANGUAGES:
        raise ValueError("{} not available. Please choose from languages {}".format(language, _AVAILABLE_LANGUAGES))
    file_name = 'news.{}.{}.shuffled'.format(year, language)
    extracted_files = [f for f in extracted_files if file_name in f]
    path = extracted_files[0]
    logging.info('Creating {} data'.format(split))
    description = construct_dataset_description("WMTNewsCrawl", root, split, year=year, language=language)
    return RawTextIterableDataset(description,
                                   NUM_LINES[split], iter(io.open(path, encoding="utf8")))
