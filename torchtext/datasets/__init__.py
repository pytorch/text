from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank  # NOQA
from .nli import SNLI, MultiNLI, XNLI
from .sst import SST
from .translation import TranslationDataset, Multi30k, IWSLT, WMT14  # NOQA
from .sequence_tagging import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking  # NOQA
from .trec import TREC
from .imdb import IMDB
from .babi import BABI20
from .unsupervised_learning import EnWik9

__all__ = ['LanguageModelingDataset',
           'SNLI',
           'MultiNLI',
           'XNLI',
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
           'TextClassificationDataset',
           'AG_NEWS',
           'SogouNews',
           'DBpedia',
           'YelpReviewPolarity',
           'YelpReviewFull',
           'YahooAnswers',
           'AmazonReviewPolarity',
           'AmazonReviewFull',
           'EnWik9']
