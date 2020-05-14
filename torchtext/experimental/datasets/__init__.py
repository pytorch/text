from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank  # NOQA
from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .question_answer import SQuAD1, SQuAD2

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
           'SQuAD1', 'SQuAD2']
