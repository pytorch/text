torchtext.datasets
==================

.. currentmodule:: torchtext.datasets


.. _datapipes_warnings:

.. warning::

    The datasets supported by torchtext are datapipes from the `torchdata
    project <https://pytorch.org/data/beta/index.html>`_, which is still in Beta
    status. This means that the API is subject to change without deprecation
    cycles. In particular, we expect a lot of the current idioms to change with
    the eventual release of ``DataLoaderV2`` from ``torchdata``.

    Here are a few recommendations regarding the use of datapipes:

    - For shuffling the datapipe, do that in the DataLoader: ``DataLoader(dp, shuffle=True)``.
      You do not need to call ``dp.shuffle()``, because ``torchtext`` has
      already done that for you. Note however that the datapipe won't be
      shuffled unless you explicitly pass ``shuffle=True`` to the DataLoader.

    - When using multi-processing (``num_workers=N``), use the builtin ``worker_init_fn``::

            from torch.utils.data.backward_compatibility import worker_init_fn
            DataLoader(dp, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)

      This will ensure that data isn't duplicated across workers.

    - We also recommend using ``drop_last=True``. Without this, the batch sizes
      at the end of an epoch may be very small in some cases (smaller than with
      other map-style datasets). This might affect accuracy greatly especially
      when batch-norm is used. ``drop_last=True`` ensures that all batch sizes
      are equal.

    - Distributed training with ``DistributedDataParallel`` is not yet entirely
      stable / supported, and we don't recommend it at this point. It will be
      better supported in DataLoaderV2. If you still wish to use DDP, make sure
      that:

      - All workers (DDP workers *and* DataLoader workers) see a different part
        of the data. The datasets are already wrapped inside  `ShardingFilter
        <https://pytorch.org/data/main/generated/torchdata.datapipes.iter.ShardingFilter.html>`_
        and you may need to call ``dp.apply_sharding(num_shards, shard_id)`` in order to shard the
        data across ranks (DDP workers) and DataLoader workers. One way to do this
        is to create ``worker_init_fn`` that calls ``apply_sharding`` with appropriate
        number of shards (DDP workers * DataLoader workers) and shard id (inferred through rank
        and worker ID of corresponding DataLoader withing rank). Note however, that this assumes
        equal number of DataLoader workers for all the ranks.
      - All DDP workers work on the same number of batches. One way to do this
        is to by limit the size of the datapipe within each worker to
        ``len(datapipe) // num_ddp_workers``, but this might not suit all
        use-cases.
      - The shuffling seed is the same across all workers. You might need to
        call ``torch.utils.data.graph_settings.apply_shuffle_seed(dp, rng)``
      - The shuffling seed is different across epochs.
      - The rest of the RNG (typically used for transformations) is
        **different** across workers, for maximal entropy and optimal accuracy.

General use cases are as follows: ::


    # import datasets
    from torchtext.datasets import IMDB

    train_iter = IMDB(split='train')

    def tokenize(label, line):
        return line.split()

    tokens = []
    for label, line in train_iter:
        tokens += tokenize(label, line)

The following datasets are currently available. If you would like to contribute
new datasets to the repo or work with your own custom datasets, please refer to `CONTRIBUTING_DATASETS.md <https://github.com/pytorch/text/blob/main/CONTRIBUTING_DATASETS.md>`_ guide.

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

CoLA
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: CoLA

DBpedia
~~~~~~~

.. autofunction:: DBpedia

IMDb
~~~~

.. autofunction:: IMDB

MNLI
~~~~

.. autofunction:: MNLI

MRPC
~~~~

.. autofunction:: MRPC

QNLI
~~~~

.. autofunction:: QNLI

QQP
~~~~

.. autofunction:: QQP

RTE
~~~~

.. autofunction:: RTE

SogouNews
~~~~~~~~~

.. autofunction:: SogouNews

SST2
~~~~

.. autofunction:: SST2

STSB
~~~~

.. autofunction:: STSB

WNLI
~~~~

.. autofunction:: WNLI

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
