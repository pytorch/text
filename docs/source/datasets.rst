torchtext.datasets
==================

.. currentmodule:: torchtext.datasets

General use cases are as follows: ::


    # import datasets
    from torchtext.datasets import IMDB

    train_iter = IMDB(split='train')

    def tokenize(label, line):
        return line.split()

    tokens = []
    for label, line in train_iter:
        tokens += tokenize(label, line)

The following datasets are available:

.. contents:: Datasets
    :local:


Text Classification
^^^^^^^^^^^^^^^^^^^

AG_NEWS
~~~~~~~

.. autofunction:: AG_NEWS

AmazonReviewFull
~~~~~~~~~~~~~~~~

.. autofunction:: AmazonReviewFull

AmazonReviewPolarity
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: AmazonReviewPolarity

DBpedia
~~~~~~~

.. autofunction:: DBpedia

IMDb
~~~~

.. autofunction:: IMDB

SogouNews
~~~~~~~~~

.. autofunction:: SogouNews

SST2
~~~~

.. autofunction:: SST2

YahooAnswers
~~~~~~~~~~~~

.. autofunction:: YahooAnswers

YelpReviewFull
~~~~~~~~~~~~~~

.. autofunction:: YelpReviewFull

YelpReviewPolarity
~~~~~~~~~~~~~~~~~~

.. autofunction:: YelpReviewPolarity


Language Modeling
^^^^^^^^^^^^^^^^^

PennTreebank
~~~~~~~~~~~~

.. autofunction:: PennTreebank

WikiText-2
~~~~~~~~~~

.. autofunction:: WikiText2

WikiText103
~~~~~~~~~~~

.. autofunction:: WikiText103


Machine Translation
^^^^^^^^^^^^^^^^^^^

IWSLT2016
~~~~~~~~~

.. autofunction:: IWSLT2016

IWSLT2017
~~~~~~~~~

.. autofunction:: IWSLT2017

Multi30k
~~~~~~~~

.. autofunction:: Multi30k


Sequence Tagging
^^^^^^^^^^^^^^^^

CoNLL2000Chunking
~~~~~~~~~~~~~~~~~

.. autofunction:: CoNLL2000Chunking

UDPOS
~~~~~

.. autofunction:: UDPOS


Question Answer
^^^^^^^^^^^^^^^

SQuAD 1.0
~~~~~~~~~

.. autofunction:: SQuAD1


SQuAD 2.0
~~~~~~~~~

.. autofunction:: SQuAD2


Unsupervised Learning
^^^^^^^^^^^^^^^^^^^^^

CC100
~~~~~~

.. autofunction:: CC100

EnWik9
~~~~~~

.. autofunction:: EnWik9
