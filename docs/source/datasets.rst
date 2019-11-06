torchtext.datasets
====================

.. currentmodule:: torchtext.datasets

General use cases are as follows: ::


    # import datasets
    from torchtext.datasets import IMDB

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

SST
~~~

.. autoclass:: SST
  :members: splits, iters

IMDb
~~~~

.. autoclass:: IMDB
  :members: __init__ 


TextClassificationDataset
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextClassificationDataset
  :members: __init__

AG_NEWS
~~~~~~~

AG_NEWS dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AG_NEWS
  :members: __init__

SogouNews
~~~~~~~~~

SogouNews dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: SogouNews
  :members: __init__

DBpedia
~~~~~~~

DBpedia dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: DBpedia
  :members: __init__

YelpReviewPolarity
~~~~~~~~~~~~~~~~~~

YelpReviewPolarity dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YelpReviewPolarity
  :members: __init__

YelpReviewFull
~~~~~~~~~~~~~~

YelpReviewFull dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YelpReviewFull
  :members: __init__

YahooAnswers
~~~~~~~~~~~~

YahooAnswers dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: YahooAnswers
  :members: __init__

AmazonReviewPolarity
~~~~~~~~~~~~~~~~~~~~

AmazonReviewPolarity dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AmazonReviewPolarity
  :members: __init__

AmazonReviewFull
~~~~~~~~~~~~~~~~

AmazonReviewFull dataset is subclass of ``TextClassificationDataset`` class.

.. autoclass:: AmazonReviewFull
  :members: __init__


Question Classification
^^^^^^^^^^^^^^^^^^^^^^^

TREC
~~~~

.. autoclass:: TREC
  :members: splits, iters

Entailment
^^^^^^^^^^

SNLI
~~~~

.. autoclass:: SNLI
  :members: splits, iters


MultiNLI
~~~~~~~~

.. autoclass:: MultiNLI
  :members: splits, iters


Language Modeling
^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``LanguageModelingDataset`` class.

.. autoclass:: LanguageModelingDataset
  :members: __init__


WikiText-2
~~~~~~~~~~

.. autoclass:: WikiText2
  :members: splits, iters


WikiText103
~~~~~~~~~~~

.. autoclass:: WikiText103
  :members: splits, iters


PennTreebank
~~~~~~~~~~~~

.. autoclass:: PennTreebank
  :members: splits, iters


Machine Translation
^^^^^^^^^^^^^^^^^^^

Machine translation datasets are subclasses of ``TranslationDataset`` class.

.. autoclass:: TranslationDataset
  :members: __init__


Multi30k
~~~~~~~~

.. autoclass:: Multi30k
  :members: splits

IWSLT
~~~~~

.. autoclass:: IWSLT
  :members: splits

WMT14
~~~~~

.. autoclass:: WMT14
  :members: splits


Sequence Tagging
^^^^^^^^^^^^^^^^

Sequence tagging datasets are subclasses of ``SequenceTaggingDataset`` class.

.. autoclass:: SequenceTaggingDataset
  :members: __init__


UDPOS
~~~~~

.. autoclass:: UDPOS
  :members: splits

CoNLL2000Chunking
~~~~~~~~~~~~~~~~~

.. autoclass:: CoNLL2000Chunking
  :members: splits

Question Answering
^^^^^^^^^^^^^^^^^^

BABI20
~~~~~~

.. autoclass:: BABI20
  :members: __init__, splits, iters

Unsupervised Learning
^^^^^^^^^^^^^^^^^^^^^

EnWik9
~~~~~~

.. autoclass:: EnWik9
  :members: __init__, __getitem__, __len__, __iter__, get_vocab
