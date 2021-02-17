from .text_classification import AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, \
    YelpReviewFull, YahooAnswers, \
    AmazonReviewPolarity, AmazonReviewFull, IMDB
from .sequence_tagging import UDPOS, CoNLL2000Chunking
from .translation import Multi30k, IWSLT, WMT14
from .language_modeling import WikiText2, WikiText103, PennTreebank, WMTNewsCrawl
from .question_answer import SQuAD1, SQuAD2

DATASETS = {'IMDB': IMDB,
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
            'IWSLT': IWSLT,
            'WMT14': WMT14,
            'WikiText2': WikiText2,
            'WikiText103': WikiText103,
            'PennTreebank': PennTreebank,
            'WMTNewsCrawl': WMTNewsCrawl,
            'SQuAD1': SQuAD1,
            'SQuAD2': SQuAD2}

from .text_classification import URLS as text_classification_URLS
from .sequence_tagging import URLS as sequence_tagging_URLS
from .translation import URLS as translation_URLS
from .language_modeling import URLS as language_modeling_URLS
from .question_answer import URLS as question_answer_URLS

URLS = text_classification_URLS
URLS.update(sequence_tagging_URLS)
URLS.update(translation_URLS)
URLS.update(language_modeling_URLS)
URLS.update(question_answer_URLS)

from .text_classification import NUM_LINES as text_classification_NUM_LINES
from .sequence_tagging import NUM_LINES as sequence_tagging_NUM_LINES
from .translation import NUM_LINES as translation_NUM_LINES
from .language_modeling import NUM_LINES as language_modeling_NUM_LINES
from .question_answer import NUM_LINES as question_answer_NUM_LINES

NUM_LINES = text_classification_NUM_LINES
NUM_LINES.update(sequence_tagging_NUM_LINES)
NUM_LINES.update(translation_NUM_LINES)
NUM_LINES.update(language_modeling_NUM_LINES)
NUM_LINES.update(question_answer_NUM_LINES)

from .text_classification import MD5 as text_classification_MD5
from .sequence_tagging import MD5 as sequence_tagging_MD5
from .translation import MD5 as translation_MD5
from .language_modeling import MD5 as language_modeling_MD5
from .question_answer import MD5 as question_answer_MD5

MD5 = text_classification_MD5
MD5.update(sequence_tagging_MD5)
MD5.update(translation_MD5)
MD5.update(language_modeling_MD5)
MD5.update(question_answer_MD5)

__all__ = sorted(list(map(str, DATASETS.keys())))
