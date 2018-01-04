torchtext.datasets
====================

.. currentmodule:: torchtext.datasets

All datasets are subclasses of :class:`torchtext.data.Dataset`, which
inherits from :class:`torch.utils.data.Dataset` i.e, they have ``split`` and
``iters`` methods implemented.

General use cases are as follows:

Approach 1, ``splits``: ::

    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=3, device=0)

Approach 2, ``iters``: ::

    # use default configurations
    train_iter, test_iter = datasets.IMDB.iters(batch_size=4)

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
  :members: splits, iters



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



Language Modeling
^^^^^^^^^^^^^^^^^

Language modeling datasets are subclasses of ``LanguageModelingDataset`` class.

.. autoclass:: LanguageModelingDataset
  :members: __init__


WikiText-2
~~~~~~~~~~

.. autoclass:: WikiText2
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
