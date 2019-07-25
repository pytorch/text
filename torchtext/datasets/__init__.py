from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank  # NOQA
from .nli import SNLI, MultiNLI
from .sst import SST
from .translation import TranslationDataset, Multi30k, IWSLT, WMT14  # NOQA
from .sequence_tagging import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking, ATIS  # NOQA
from .trec import TREC
from .imdb import IMDB
from .babi import BABI20
from .text_classification import TextClassificationDataset, \
    AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull


__all__ = ['LanguageModelingDataset',
           'SNLI',
           'MultiNLI',
           'SST',
           'TranslationDataset',
           'Multi30k',
           'IWSLT',
           'WMT14',
           'WikiText2',
           'WikiText103',
           'PennTreebank',
           'TREC',
           'IMDB',
           'SequenceTaggingDataset',
           'UDPOS',
           'CoNLL2000Chunking',
           'BABI20',
           'ATIS',
           'TextClassificationDataset',
           'AG_NEWS',
           'SogouNews',
           'DBpedia',
           'YelpReviewPolarity',
           'YelpReviewFull',
           'YahooAnswers',
           'AmazonReviewPolarity',
           'AmazonReviewFull']
