from .language_modeling import LanguageModelingDataset, WikiText2
from .snli import SNLI
from .sentiment import SST
from .translation import TranslationDataset
from .trec import TREC


__all__ = ['LanguageModelingDataset',
           'SNLI',
           'SST',
           'TranslationDataset',
           'WikiText2',
           'TREC']
