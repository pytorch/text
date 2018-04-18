from .language_modeling import LanguageModelingDataset, WikiText2, PennTreebank  # NOQA
from .snli import SNLI
from .sst import SST
from .translation import TranslationDataset, Multi30k, IWSLT, WMT14  # NOQA
from .sequence_tagging import SequenceTaggingDataset, UDPOS # NOQA
from .trec import TREC
from .imdb import IMDB
from .babi import BABI20


__all__ = ['LanguageModelingDataset',
           'SNLI',
           'SST',
           'TranslationDataset',
           'Multi30k',
           'IWSLT',
           'WMT14'
           'WikiText2',
           'PennTreeBank',
           'TREC',
           'IMDB',
           'SequenceTaggingDataset',
           'UDPOS',
           'BABI20']
