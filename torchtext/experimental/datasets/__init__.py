from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank  # NOQA
from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB

# Raw text
from .raw_text_classification import RawAG_NEWS, RawSogouNews, RawDBpedia, \
    RawYelpReviewPolarity, RawYelpReviewFull, RawYahooAnswers, \
    RawAmazonReviewPolarity, RawAmazonReviewFull, RawIMDB

__all__ = ['LanguageModelingDataset',
           'WikiText2',
           'WikiText103',
           'PennTreebank',
           'IMDB',
           'AG_NEWS',
           'SogouNews',
           'DBpedia',
           'YelpReviewPolarity',
           'YelpReviewFull',
           'YahooAnswers',
           'AmazonReviewPolarity',
           'AmazonReviewFull',
           'RawAG_NEWS',
           'RawSogouNews',
           'RawDBpedia',
           'RawYelpReviewPolarity',
           'RawYelpReviewFull',
           'RawYahooAnswers',
           'RawAmazonReviewPolarity',
           'RawAmazonReviewFull',
           'RawIMDB']
