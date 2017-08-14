from .language_modeling import LanguageModelingDataset, WikiText2
from .snli import SNLI
from .sst import SST
from .translation import TranslationDataset
from .trec import TREC
from .imdb import IMDB


__all__ = ['LanguageModelingDataset',
           'SNLI',
           'SST',
           'TranslationDataset',
           'WikiText2',
           'TREC',
           'IMDB']
