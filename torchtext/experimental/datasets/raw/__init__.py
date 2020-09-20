from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .sequence_tagging import UDPOS, CoNLL2000Chunking
from .translation import Multi30k, IWSLT, WMT14
from .language_modeling import WikiText2, WikiText103, PennTreebank, WMTNewsCrawl
from .question_answer import SQuAD1, SQuAD2

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

__all__ = list(map(str, DATASETS.keys()))
