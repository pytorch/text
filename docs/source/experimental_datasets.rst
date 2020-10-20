torchtext.experimental.datasets
===============================

.. currentmodule:: torchtext.experimental.datasets

The following datasets have been rewritten and more compatible with ``torch.utils.data``. General use cases are as follows: ::


    # import datasets
    from torchtext.experimental.datasets import IMDB

    # set up tokenizer (the default on is basic_english tokenizer)
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("spacy")

    # obtain data and vocab with a custom tokenizer
    train_dataset, test_dataset = IMDB(tokenizer=tokenizer)
    vocab = train_dataset.get_vocab()

    # use the default tokenizer
    train_dataset, test_dataset = IMDB()
    vocab = train_dataset.get_vocab()

The following datasets are available:

.. contents:: Datasets
    :local:


Text Classification
^^^^^^^^^^^^^^^^^^^

TextClassificationDataset
~~~~~~~~~~~~~~~~~~~~~~~~~

Text classification datasets are subclasses of ``TextClassificationDataset`` class.

.. autoclass:: TextClassificationDataset
  :members: __init__

AG_NEWS
~~~~~~~

.. autofunction:: AG_NEWS


SogouNews
~~~~~~~~~

.. autofunction:: SogouNews

DBpedia
~~~~~~~

.. autofunction:: DBpedia

YelpReviewPolarity
~~~~~~~~~~~~~~~~~~

.. autofunction:: YelpReviewPolarity

YelpReviewFull
~~~~~~~~~~~~~~

.. autofunction:: YelpReviewFull

YahooAnswers
~~~~~~~~~~~~

.. autofunction:: YahooAnswers

AmazonReviewPolarity
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: AmazonReviewPolarity

AmazonReviewFull
~~~~~~~~~~~~~~~~

.. autofunction:: AmazonReviewFull

IMDb
~~~~

.. autofunction:: IMDB


Language Modeling
^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``LanguageModelingDataset`` class.

.. autoclass:: LanguageModelingDataset
  :members: __init__


WikiText-2
~~~~~~~~~~

.. autofunction:: WikiText2


WikiText103
~~~~~~~~~~~

.. autofunction:: WikiText103


PennTreebank
~~~~~~~~~~~~

.. autofunction:: PennTreebank


WMTNewsCrawl
~~~~~~~~~~~~

.. autofunction:: WMTNewsCrawl


Machine Translation
^^^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``TranslationDataset`` class.

.. autoclass:: TranslationDataset
  :members: __init__


Multi30k
~~~~~~~~

.. autofunction:: Multi30k


IWSLT
~~~~~

.. autofunction:: IWSLT


WMT14
~~~~~

.. autofunction:: WMT14


Sequence Tagging
^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``SequenceTaggingDataset`` class.

.. autoclass:: SequenceTaggingDataset
  :members: __init__

UDPOS
~~~~~

.. autofunction:: UDPOS

CoNLL2000Chunking
~~~~~~~~~~~~~~~~~

.. autofunction:: CoNLL2000Chunking

Question Answer
^^^^^^^^^^^^^^^

Question answer datasets are subclasses of ``QuestionAnswerDataset`` class.

.. autoclass:: QuestionAnswerDataset
  :members: __init__


SQuAD 1.0
~~~~~~~~~

.. autofunction:: SQuAD1


SQuAD 2.0
~~~~~~~~~

.. autofunction:: SQuAD2
