from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .translation import Multi30k, IWSLT, WMT14
from .language_modeling import WikiText2, WikiText103, PennTreebank, WMTNewsCrawl
from .question_answer import SQuAD1, SQuAD2

__all__ = ['IMDB',
           'AG_NEWS',
           'SogouNews',
           'DBpedia',
           'YelpReviewPolarity',
           'YelpReviewFull',
           'YahooAnswers',
           'AmazonReviewPolarity',
           'AmazonReviewFull',
           'Multi30k',
           'IWSLT',
           'WMT14',
           'WikiText2',
           'WikiText103',
           'PennTreebank',
           'WMTNewsCrawl',
           'SQuAD1', 'SQuAD2']
