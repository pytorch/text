torchtext.experimental.datasets
================================

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


Sentiment Analysis
^^^^^^^^^^^^^^^^^^

IMDb
~~~~

.. autoclass:: IMDB
  :members: __init__


Text Classification
^^^^^^^^^^^^^^^^^^^

TextClassificationDataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextClassificationDataset
  :members: __init__

AG_NEWS
~~~~~~

AG_NEWS dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AG_NEWS
  :members: __init__

SogouNews
~~~~~~~~

SogouNews dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: SogouNews
  :members: __init__

DBpedia
~~~~~~

DBpedia dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: DBpedia
  :members: __init__

YelpReviewPolarity
~~~~~~~~~~~~~~~~~

YelpReviewPolarity dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YelpReviewPolarity
  :members: __init__

YelpReviewFull
~~~~~~~~~~~~~

YelpReviewFull dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YelpReviewFull
  :members: __init__

YahooAnswers
~~~~~~~~~~~

YahooAnswers dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YahooAnswers
  :members: __init__

AmazonReviewPolarity
~~~~~~~~~~~~~~~~~~~

AmazonReviewPolarity dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AmazonReviewPolarity
  :members: __init__

AmazonReviewFull
~~~~~~~~~~~~~~~

AmazonReviewFull dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AmazonReviewFull
  :members: __init__


Language Modeling
^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``LanguageModelingDataset`` class.

.. autoclass:: LanguageModelingDataset
  :members: __init__


WikiText-2
~~~~~~~~~~

.. autoclass:: WikiText2
  :members: __init__


WikiText103
~~~~~~~~~~~

.. autoclass:: WikiText103
  :members: __init__


PennTreebank
~~~~~~~~~~~~

.. autoclass:: PennTreebank
  :members: __init__


Machine Translation
^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``TranslationDataset`` class.

.. autoclass:: TranslationDataset
  :members: __init__


Multi30k
~~~~~~~~

.. autoclass:: Multi30k
  :members: __init__


IWSLT
~~~~~

.. autoclass:: IWSLT
  :members: __init__


WMT14
~~~~~

.. autoclass:: WMT14
  :members: __init__
