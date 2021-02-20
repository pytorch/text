from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import download_extract_validate
import io
import logging

URL = 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'

MD5 = 'c70da2ba79db33fb0fc9119cbad16260'

NUM_LINES = {
    'train': 17676013,
}

_PATH = "training-monolingual-news-2010.tgz"

_AVAILABLE_YEARS = [2010]
_AVAILABLE_LANGUAGES = [
    "cs",
    "en",
    "fr",
    "es",
    "de"
]

_EXTRACTED_FILES = {
    "cs": "training-monolingual/news.2010.cs.shuffled",
    "de": "training-monolingual/news.2010.de.shuffled",
    "en": "training-monolingual/news.2010.en.shuffled",
    "es": "training-monolingual/news.2010.es.shuffled",
    "fr": "training-monolingual/news.2010.fr.shuffled"
}

_EXTRACTED_FILES_MD5 = {
    "cs": "b60fdbf95a2e97bae3a7d04cc81df925",
    "de": "e59a0f0c6eeeb2113c0da1873a2e1035",
    "en": "234a50914d87158754815a0bd86d7b9d",
    "es": "aee3d773a0c054c5ac313a42d08b7020",
    "fr": "066d671533f78bfe139cf7052574fd5a"
}


@add_docstring_header()
@wrap_split_argument(('train',))
def WMTNewsCrawl(root, split, year=2010, language='en'):
    if year not in _AVAILABLE_YEARS:
        raise ValueError("{} not available. Please choose from years {}".format(year, _AVAILABLE_YEARS))
    if language not in _AVAILABLE_LANGUAGES:
        raise ValueError("{} not available. Please choose from languages {}".format(language, _AVAILABLE_LANGUAGES))
    path = download_extract_validate(root, URL, MD5, _PATH, _EXTRACTED_FILES[language],
                                     _EXTRACTED_FILES_MD5[language], hash_type="md5")
    logging.info('Creating {} data'.format(split))
    return RawTextIterableDataset("WMTNewsCrawl",
                                  NUM_LINES[split], iter(io.open(path, encoding="utf8")))
