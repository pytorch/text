from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank, WMTNewsCrawl  # NOQA
from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .sequence_tagging import UDPOS, CoNLL2000Chunking
from .translation import Multi30k, IWSLT, WMT14
from .question_answer import SQuAD1, SQuAD2

__all__ = ['LanguageModelingDataset',
           'WikiText2',
           'WikiText103',
           'PennTreebank',
           'WMTNewsCrawl',
           'IMDB',
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
           'SQuAD1', 'SQuAD2']
