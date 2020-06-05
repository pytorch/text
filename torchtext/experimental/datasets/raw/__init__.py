from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .sequence_tagging import UDPOS, CoNLL2000Chunking
from .translation import Multi30k, IWSLT, WMT14
from .language_modeling import WikiText2, WikiText103, PennTreebank, WMTNewsCrawl

__all__ = ['IMDB',
           'AG_NEWS',
           'SogouNews',
           'DBpedia',
           'YelpReviewPolarity',
           'YelpReviewFull',
           'YahooAnswers',
           'AmazonReviewPolarity',
           'AmazonReviewFull',
           'UDPOS',
           'CoNLL2000Chunking',
           'Multi30k',
           'IWSLT',
           'WMT14',
           'WikiText2',
           'WikiText103',
           'PennTreebank',
           'WMTNewsCrawl']
