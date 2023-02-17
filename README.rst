.. image:: docs/source/_static/img/torchtext_logo.png

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
* `torchtext.data <https://github.com/pytorch/text/tree/main/torchtext/data>`_: Some basic NLP building blocks
* `torchtext.transforms <https://github.com/pytorch/text/tree/main/torchtext/transforms>`_: Basic text-processing transformations
* `torchtext.models <https://github.com/pytorch/text/tree/main/torchtext/models>`_: Pre-trained models
* `torchtext.vocab <https://github.com/pytorch/text/tree/main/torchtext/vocab>`_: Vocab and Vectors related classes and factory functions
* `examples <https://github.com/pytorch/text/tree/main/examples>`_: Example NLP workflows with PyTorch and torchtext library.


Installation
============

We recommend Anaconda as a Python package management system. Please refer to `pytorch.org <https://pytorch.org/>`_ for the details of PyTorch installation. The following are the corresponding ``torchtext`` versions and supported Python versions.

.. csv-table:: Version Compatibility
   :header: "PyTorch version", "torchtext version", "Supported Python version"
   :widths: 10, 10, 10

   nightly build, main, ">=3.8, <=3.11"
   1.13.0, 0.14.0, ">=3.7, <=3.10"
   1.12.0, 0.13.0, ">=3.7, <=3.10"
   1.11.0, 0.12.0, ">=3.6, <=3.9"
   1.10.0, 0.11.0, ">=3.6, <=3.9"
   1.9.1, 0.10.1, ">=3.6, <=3.9"
   1.9, 0.10, ">=3.6, <=3.9"
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
    CC=clang CXX=clang++ python setup.py clean install

    # or ``python setup.py develop`` if you are making modifications.

**Note**

When building from source, make sure that you have the same C++ compiler as the one used to build PyTorch. A simple way is to build PyTorch from source and use the same environment to build torchtext.
If you are using the nightly build of PyTorch, checkout the environment it was built with `conda (here) <https://github.com/pytorch/builder/tree/main/conda>`_ and `pip (here) <https://github.com/pytorch/builder/tree/main/manywheel>`_.

Additionally, datasets in torchtext are implemented using the torchdata library. Please take a look at the
`installation instructions <https://github.com/pytorch/data#installation>`_ to download the latest nightlies or install from source.

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
* Text classification: SST2, AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, YelpReviewFull, YahooAnswers, AmazonReviewPolarity, AmazonReviewFull, IMDB
* Model pre-training: CC-100

Models
======

The library currently consist of following pre-trained models:

* RoBERTa: `Base and Large Architecture <https://github.com/pytorch/fairseq/tree/main/examples/roberta#pre-trained-models>`_
* `DistilRoBERTa <https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/README.md>`_
* XLM-RoBERTa: `Base and Large Architure <https://github.com/pytorch/fairseq/tree/main/examples/xlmr#pre-trained-models>`_

Tokenizers
==========

The transforms module currently support following scriptable tokenizers:

* `SentencePiece <https://github.com/google/sentencepiece>`_
* `GPT-2 BPE <https://github.com/openai/gpt-2/blob/master/src/encoder.py>`_
* `CLIP <https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py>`_

Tutorials
=========

To get started with torchtext, users may refer to the following tutorial available on PyTorch website.

* `SST-2 binary text classification using XLM-R pre-trained model <https://pytorch.org/text/stable/tutorials/sst2_classification_non_distributed.html>`_
* `Text classification with AG_NEWS dataset <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html>`_
* `Translation trained with Multi30k dataset using transformers and torchtext <https://pytorch.org/tutorials/beginner/translation_transformer.html>`_
* `Language modeling using transforms and torchtext <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_


Disclaimer on Datasets
======================

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
