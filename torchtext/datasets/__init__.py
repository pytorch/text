from .language_modeling import LanguageModelingDataset, WikiText2
from .snli import SNLI
from .sst import SST
from .translation import TranslationDataset, Multi30k, IWSLT, WMT14
from .trec import TREC
from .imdb import IMDB


__all__ = ['LanguageModelingDataset',
           'SNLI',
           'SST',
           'TranslationDataset',
           'Multi30k',
           'IWSLT',
           'WMT14'
           'WikiText2',
           'TREC',
           'IMDB']
