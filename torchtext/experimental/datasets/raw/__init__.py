import importlib
from .ag_news import AG_NEWS
from .sogounews import SogouNews
from .dbpedia import DBpedia
from .yelpreviewpolarity import YelpReviewPolarity
from .yelpreviewfull import YelpReviewFull
from .yahooanswers import YahooAnswers
from .amazonreviewpolarity import AmazonReviewPolarity
from .amazonreviewfull import AmazonReviewFull
from .imdb import IMDB

from .wikitext2 import WikiText2
from .wikitext103 import WikiText103
from .penntreebank import PennTreebank
from .wmtnewscrawl import WMTNewsCrawl

from .squad1 import SQuAD1
from .squad2 import SQuAD2

from .sequence_tagging import UDPOS, CoNLL2000Chunking

from .multi30k import Multi30k
from .iwslt import IWSLT
from .wmt14 import WMT14

DATASETS = {'IMDB': IMDB,
            'AG_NEWS': AG_NEWS,
            'SogouNews': SogouNews,
            'DBpedia': DBpedia,
            'YelpReviewPolarity': YelpReviewPolarity,
            'YelpReviewFull': YelpReviewFull,
            'YahooAnswers': YahooAnswers,
            'AmazonReviewPolarity': AmazonReviewPolarity,
            'AmazonReviewFull': AmazonReviewFull,
            'UDPOS': UDPOS,
            'CoNLL2000Chunking': CoNLL2000Chunking,
            'Multi30k': Multi30k,
            'IWSLT': IWSLT,
            'WMT14': WMT14,
            'WikiText2': WikiText2,
            'WikiText103': WikiText103,
            'PennTreebank': PennTreebank,
            'WMTNewsCrawl': WMTNewsCrawl,
            'SQuAD1': SQuAD1,
            'SQuAD2': SQuAD2}

URLS = {}
NUM_LINES = {}
MD5 = {}
for dataset in ["AG_NEWS",
                "SogouNews",
                "DBpedia",
                "YelpReviewPolarity",
                "YelpReviewFull",
                "YahooAnswers",
                "AmazonReviewPolarity",
                "AmazonReviewFull",
                "IMDB",
                "WikiText2",
                "WikiText103",
                "PennTreebank",
                "WMTNewsCrawl",
                "SQuAD1",
<<<<<<< HEAD
                "SQuAD2",
                "Multi30k",
                "IWSLT",
                "WMT14"]:
=======
                "SQuAD2"]:
>>>>>>> 71065192f529a886b27704ef5e7a3d2c700cba7b
    dataset_module_path = "torchtext.experimental.datasets.raw." + dataset.lower()
    dataset_module = importlib.import_module(dataset_module_path)
    URLS[dataset] = dataset_module.URL
    NUM_LINES[dataset] = dataset_module.NUM_LINES
    MD5[dataset] = dataset_module.MD5

from .sequence_tagging import URLS as sequence_tagging_URLS
from .translation import URLS as translation_URLS

URLS.update(sequence_tagging_URLS)
URLS.update(translation_URLS)

from .sequence_tagging import NUM_LINES as sequence_tagging_NUM_LINES
from .translation import NUM_LINES as translation_NUM_LINES

NUM_LINES.update(sequence_tagging_NUM_LINES)
NUM_LINES.update(translation_NUM_LINES)

from .sequence_tagging import MD5 as sequence_tagging_MD5
from .translation import MD5 as translation_MD5

MD5.update(sequence_tagging_MD5)
MD5.update(translation_MD5)

__all__ = sorted(list(map(str, DATASETS.keys())))
