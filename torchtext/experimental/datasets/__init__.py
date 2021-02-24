from .language_modeling import LanguageModelingDataset, WikiText2, WikiText103, PennTreebank, WMTNewsCrawl  # NOQA: F401
from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .text_classification import TextClassificationDataset  # NOQA: F401
from .sequence_tagging import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking  # NOQA: F401
from .translation import TranslationDataset, Multi30k, IWSLT2016, IWSLT2017, WMT14  # NOQA: F401
from .question_answer import QuestionAnswerDataset, SQuAD1, SQuAD2  # NOQA: F401


DATASETS = {'WikiText2': WikiText2,
            'WikiText103': WikiText103,
            'PennTreebank': PennTreebank,
            'WMTNewsCrawl': WMTNewsCrawl,
            'IMDB': IMDB,
            'AG_NEWS': AG_NEWS,
            'SogouNews': SogouNews,
            'DBpedia': DBpedia,
            'YelpReviewPolarity': YelpReviewPolarity,
            'YelpReviewFull': YelpReviewFull,
            'YahooAnswers': YahooAnswers,
            'AmazonReviewPolarity': AmazonReviewPolarity,
            'AmazonReviewFull': AmazonReviewFull,
            'UDPOS': UDPOS,
            'CoNLL2000Chunking': CoNLL2000Chunking,
            'Multi30k': Multi30k,
            'IWSLT2016': IWSLT2016,
            'IWSLT2017': IWSLT2017,
            'WMT14': WMT14,
            'SQuAD1': SQuAD1,
            'SQuAD2': SQuAD2}

__all__ = sorted(list(map(str, DATASETS.keys())))
