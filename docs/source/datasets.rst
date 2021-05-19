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

WikiText-2
~~~~~~~~~~

.. autofunction:: WikiText2


WikiText103
~~~~~~~~~~~

.. autofunction:: WikiText103


PennTreebank
~~~~~~~~~~~~

.. autofunction:: PennTreebank


Machine Translation
^^^^^^^^^^^^^^^^^^^

Multi30k
~~~~~~~~

.. autofunction:: Multi30k



IWSLT2016
~~~~~~~~~

.. autofunction:: IWSLT2016

IWSLT2017
~~~~~~~~~

.. autofunction:: IWSLT2017


Sequence Tagging
^^^^^^^^^^^^^^^^

UDPOS
~~~~~

.. autofunction:: UDPOS

CoNLL2000Chunking
~~~~~~~~~~~~~~~~~

.. autofunction:: CoNLL2000Chunking


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

EnWik9
~~~~~~

.. autofunction:: EnWik9

