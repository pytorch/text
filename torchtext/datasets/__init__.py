import importlib

from .ag_news import AG_NEWS
from .amazonreviewfull import AmazonReviewFull
from .amazonreviewpolarity import AmazonReviewPolarity
from .cc100 import CC100
from .cnndm import CNNDM
from .cola import CoLA
from .conll2000chunking import CoNLL2000Chunking
from .dbpedia import DBpedia
from .enwik9 import EnWik9
from .imdb import IMDB
from .iwslt2016 import IWSLT2016
from .iwslt2017 import IWSLT2017
from .mnli import MNLI
from .mrpc import MRPC
from .multi30k import Multi30k
from .penntreebank import PennTreebank
from .qnli import QNLI
from .qqp import QQP
from .rte import RTE
from .sogounews import SogouNews
from .squad1 import SQuAD1
from .squad2 import SQuAD2
from .sst2 import SST2
from .stsb import STSB
from .udpos import UDPOS
from .wikitext103 import WikiText103
from .wikitext2 import WikiText2
from .wnli import WNLI
from .yahooanswers import YahooAnswers
from .yelpreviewfull import YelpReviewFull
from .yelpreviewpolarity import YelpReviewPolarity

DATASETS = {
    "AG_NEWS": AG_NEWS,
    "AmazonReviewFull": AmazonReviewFull,
    "AmazonReviewPolarity": AmazonReviewPolarity,
    "CC100": CC100,
    "CoLA": CoLA,
    "CoNLL2000Chunking": CoNLL2000Chunking,
    "DBpedia": DBpedia,
    "EnWik9": EnWik9,
    "IMDB": IMDB,
    "IWSLT2016": IWSLT2016,
    "IWSLT2017": IWSLT2017,
    "MNLI": MNLI,
    "MRPC": MRPC,
    "Multi30k": Multi30k,
    "PennTreebank": PennTreebank,
    "QNLI": QNLI,
    "QQP": QQP,
    "RTE": RTE,
    "SQuAD1": SQuAD1,
    "SQuAD2": SQuAD2,
    "SogouNews": SogouNews,
    "SST2": SST2,
    "STSB": STSB,
    "UDPOS": UDPOS,
    "WikiText103": WikiText103,
    "WikiText2": WikiText2,
    "WNLI": WNLI,
    "YahooAnswers": YahooAnswers,
    "YelpReviewFull": YelpReviewFull,
    "YelpReviewPolarity": YelpReviewPolarity,
    "CNNDM": CNNDM,
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
