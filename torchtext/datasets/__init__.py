from .language_modeling import LanguageModelingDataset, WikiText2, PennTreebank  # NOQA
from .nli import SNLI, MultiNLI
from .sst import SST
from .translation import TranslationDataset, Multi30k, IWSLT, WMT14  # NOQA
from .sequence_tagging import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking # NOQA
from .trec import TREC
from .imdb import IMDB
from .babi import BABI20


__all__ = ['LanguageModelingDataset',
           'SNLI',
           'MultiNLI',
           'SST',
           'TranslationDataset',
           'Multi30k',
           'IWSLT',
           'WMT14',
           'WikiText2',
           'PennTreeBank',
           'TREC',
           'IMDB',
           'SequenceTaggingDataset',
           'UDPOS',
           'CoNLL2000Chunking',
           'BABI20']
