.. image:: https://circleci.com/gh/pytorch/text.svg?style=svg
    :target: https://circleci.com/gh/pytorch/text

.. image:: https://codecov.io/gh/pytorch/text/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/pytorch/text

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchtext%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v
    :target: https://pytorch.org/text/

torchtext
+++++++++

This repository consists of:

* `torchtext.datasets <https://github.com/pytorch/text/tree/main/torchtext/datasets>`_: The raw text iterators for common NLP datasets
* `torchtext.data <https://github.com/pytorch/text/tree/main/torchtext/data>`_: Some basic NLP building blocks (tokenizers, metrics, functionals etc.)
* `torchtext.nn <https://github.com/pytorch/text/tree/main/torchtext/nn>`_: NLP related modules
* `torchtext.vocab <https://github.com/pytorch/text/tree/main/torchtext/vocab>`_: Vocab and Vectors related classes and factory functions
* `examples <https://github.com/pytorch/text/tree/main/examples>`_: Example NLP workflows with PyTorch and torchtext library.

Note: The legacy code discussed in `torchtext v0.7.0 release note <https://github.com/pytorch/text/releases/tag/v0.7.0-rc3>`_ has been retired to `torchtext.legacy <https://github.com/pytorch/text/tree/release/0.9/torchtext/legacy>`_ folder. Those legacy code will not be maintained by the development team, and we plan to fully remove them in the future release. See `torchtext.legacy <https://github.com/pytorch/text/tree/release/0.9/torchtext/legacy>`_ folder for more details.

Installation
============

We recommend Anaconda as a Python package management system. Please refer to `pytorch.org <https://pytorch.org/>`_ for the details of PyTorch installation. The following are the corresponding ``torchtext`` versions and supported Python versions.

.. csv-table:: Version Compatibility
   :header: "PyTorch version", "torchtext version", "Supported Python version"
   :widths: 10, 10, 10

   nightly build, main, ">=3.7, <=3.9"
   1.10.0, 0.11.0, ">=3.6, <=3.9" 
   1.9.1, 0.10.1, ">=3.6, <=3.9" 
   1.9, 0.10, ">=3.6, <=3.9"
   1.8.2, 0.9.2, ">=3.6, <=3.9"
   1.8.1, 0.9.1, ">=3.6, <=3.9"
   1.8, 0.9, ">=3.6, <=3.9"
   1.7.1, 0.8.1, ">=3.6, <=3.9"
   1.7, 0.8, ">=3.6, <=3.8"
   1.6, 0.7, ">=3.6, <=3.8"
   1.5, 0.6, ">=3.5, <=3.8"
   1.4, 0.5, "2.7, >=3.5, <=3.8"
   0.4 and below, 0.2.3, "2.7, >=3.5, <=3.8"

Using conda::

    conda install -c pytorch torchtext

Using pip::

    pip install torchtext

Optional requirements
---------------------

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`_, you need to install SpaCy and download its English model::

    pip install spacy
    python -m spacy download en_core_web_sm 

Alternatively, you might want to use the `Moses <http://www.statmt.org/moses/>`_ tokenizer port in `SacreMoses <https://github.com/alvations/sacremoses>`_ (split from `NLTK <http://nltk.org/>`_). You have to install SacreMoses::

    pip install sacremoses

For torchtext 0.5 and below, ``sentencepiece``::

    conda install -c powerai sentencepiece

Building from source
--------------------

To build torchtext from source, you need ``git``, ``CMake`` and C++11 compiler such as ``g++``.::

    git clone https://github.com/pytorch/text torchtext
    cd torchtext
    git submodule update --init --recursive

    # Linux
    python setup.py clean install

    # OSX
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py clean install

    # or ``python setup.py develop`` if you are making modifications.

**Note**

When building from source, make sure that you have the same C++ compiler as the one used to build PyTorch. A simple way is to build PyTorch from source and use the same environment to build torchtext.
If you are using the nightly build of PyTorch, checkout the environment it was built with `conda (here) <https://github.com/pytorch/builder/tree/main/conda>`_ and `pip (here) <https://github.com/pytorch/builder/tree/main/manywheel>`_.

Documentation
=============

Find the documentation `here <https://pytorch.org/text/>`_.

Datasets
========

The datasets module currently contains:

* Language modeling: WikiText2, WikiText103, PennTreebank, EnWik9
* Machine translation: IWSLT2016, IWSLT2017, Multi30k
* Sequence tagging (e.g. POS/NER): UDPOS, CoNLL2000Chunking
* Question answering: SQuAD1, SQuAD2 
* Text classification: AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, YelpReviewFull, YahooAnswers, AmazonReviewPolarity, AmazonReviewFull, IMDB

For example, to access the raw text from the AG_NEWS dataset:

  .. code-block:: python

      >>> from torchtext.datasets import AG_NEWS
      >>> train_iter = AG_NEWS(split='train')
      >>> # Iterate with for loop
      >>> for (label, line) in train_iter:
      >>>     print(label, line)
      >>> # Or send to DataLoader
      >>> from torch.utils.data import DataLoader
      >>> train_iter = AG_NEWS(split='train')
      >>> dataloader = DataLoader(train_iter, batch_size=8, shuffle=False)

Tutorials
=========

To get started with torchtext, users may refer to the following tutorials available on PyTorch website.

* `Text classification with AG_NEWS dataset <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html>`_
* `Translation trained with Multi30k dataset using transformers and torchtext <https://pytorch.org/tutorials/beginner/translation_transformer.html>`_
* `Language modeling using transforms and torchtext <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_


[BC Breaking] Legacy
====================

In the v0.9.0 release, we moved the following legacy code to `torchtext.legacy <https://github.com/pytorch/text/tree/release/0.9/torchtext/legacy>`_. This is part of the work to revamp the torchtext library and the motivation has been discussed in `Issue #664 <https://github.com/pytorch/text/issues/664>`_:

* ``torchtext.legacy.data.field``
* ``torchtext.legacy.data.batch``
* ``torchtext.legacy.data.example``
* ``torchtext.legacy.data.iterator``
* ``torchtext.legacy.data.pipeline``
* ``torchtext.legacy.datasets``

We have a `migration tutorial <https://colab.research.google.com/github/pytorch/text/blob/release/0.9/examples/legacy_tutorial/migration_tutorial.ipynb>`_ to help users switch to the torchtext datasets in ``v0.9.0`` release. For the users who still want the legacy components, they can add ``legacy`` to the import path.  

In the v0.10.0 release, we retire the Vocab class to `torchtext.legacy <https://github.com/pytorch/text/tree/release/0.9/torchtext/legacy>`_. Users can still access the legacy Vocab via ``torchtext.legacy.vocab``. This class has been replaced by a Vocab module that is backed by efficient C++ implementation and provides common functional APIs for NLP workflows. 

Disclaimer on Datasets
======================

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
