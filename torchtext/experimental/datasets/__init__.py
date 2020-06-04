from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank, WMTNewsCrawl  # NOQA
from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB

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
           'AmazonReviewFull']
