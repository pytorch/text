import importlib
from .ag_news import AG_NEWS
from .amazonreviewfull import AmazonReviewFull
from .amazonreviewpolarity import AmazonReviewPolarity
from .conll2000chunking import CoNLL2000Chunking
from .dbpedia import DBpedia
from .enwik9 import EnWik9
from .imdb import IMDB
from .iwslt import IWSLT
from .matchedmultinli import MatchedMultiNLI
from .multi30k import Multi30k
from .penntreebank import PennTreebank
from .snli import SNLI
from .sogounews import SogouNews
from .squad1 import SQuAD1
from .squad2 import SQuAD2
from .udpos import UDPOS
from .wikitext103 import WikiText103
from .wikitext2 import WikiText2
from .wmt14 import WMT14
from .wmtnewscrawl import WMTNewsCrawl
from .yahooanswers import YahooAnswers
from .yelpreviewfull import YelpReviewFull
from .yelpreviewpolarity import YelpReviewPolarity

DATASETS = {
    'AG_NEWS': AG_NEWS,
    'AmazonReviewFull': AmazonReviewFull,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'CoNLL2000Chunking': CoNLL2000Chunking,
    'DBpedia': DBpedia,
    'EnWik9': EnWik9,
    'IMDB': IMDB,
    'IWSLT': IWSLT,
    'MatchedMultiNLI': MatchedMultiNLI,
    'Multi30k': Multi30k,
    'PennTreebank': PennTreebank,
    'SNLI': SNLI,
    'SQuAD1': SQuAD1,
    'SQuAD2': SQuAD2,
    'SogouNews': SogouNews,
    'UDPOS': UDPOS,
    'WMT14': WMT14,
    'WMTNewsCrawl': WMTNewsCrawl,
    'WikiText103': WikiText103,
    'WikiText2': WikiText2,
    'YahooAnswers': YahooAnswers,
    'YelpReviewFull': YelpReviewFull,
    'YelpReviewPolarity': YelpReviewPolarity
}

URLS = {}
NUM_LINES = {}
MD5 = {}
for dataset in DATASETS:
    dataset_module_path = "torchtext.datasets." + dataset.lower()
    dataset_module = importlib.import_module(dataset_module_path)
    URLS[dataset] = dataset_module.URL
    NUM_LINES[dataset] = dataset_module.NUM_LINES
    MD5[dataset] = dataset_module.MD5

__all__ = sorted(list(map(str, DATASETS.keys())))
