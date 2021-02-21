.. image:: https://circleci.com/gh/pytorch/text.svg?style=svg
    :target: https://circleci.com/gh/pytorch/text

.. image:: https://codecov.io/gh/pytorch/text/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pytorch/text

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchtext%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v
    :target: https://pytorch.org/text/

torchtext
+++++++++

This repository consists of:

* `torchtext.datasets <#datasets>`_: The raw text iterators for common NLP datasets
* `torchtext.data <#data>`_: Some basic NLP building blocks (tokenizers, metrics, functionals etc.)
* `torchtext.nn <#nn>`_: NLP related modules
* `examples <https://github.com/pytorch/text/tree/master/examples>`_: Example NLP workflows with PyTorch and torchtext library.

Note: the legacy code discussed in `torchtext v0.7.0 release note <https://github.com/pytorch/text/releases/tag/v0.7.0-rc3>`_ has been retired to `torchtext.legacy <#legacy>`_ folder. Those legacy code will not be maintained by the development team and we plan to fully remove then in the future release. See the Legacy session for more details.

Installation
============

We recommend Anaconda as Python package management system. Please refer to `pytorch.org <https://pytorch.org/>`_ for the detail of PyTorch installation. The following is the corresponding ``torchtext`` versions and supported Python versions.

.. csv-table:: Version Compatibility
   :header: "PyTorch version", "torchtext version", "Supported Python version"
   :widths: 10, 10, 10

   nightly build, master, 3.6+
   1.8, 0.9, 3.6+
   1.7, 0.8, 3.6+
   1.6, 0.7, 3.6+
   1.5, 0.6, 3.5+
   1.4, 0.5, "2.7, 3.5+"
   0.4 and below, 0.2.3, "2.7, 3.5+"

Using conda::

    conda install -c pytorch torchtext

Using pip::

    pip install torchtext

Optional requirements
---------------------

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`_, you need to install SpaCy and download its English model::

    pip install spacy
    python -m spacy download en

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
If you are using the nightly build of PyTorch, checkout the environment it was built with `conda (here) <https://github.com/pytorch/builder/tree/master/conda>`_ and `pip (here) <https://github.com/pytorch/builder/tree/master/manywheel>`_.

Documentation
=============

Find the documentation `here <https://pytorch.org/text/>`_.

Datasets
========

The datasets module currently contains:

* Language modeling: WikiText2, WikiText103, PennTreebank, EnWik9
* Machine translation: Multi30k, IWSLT, WMT14
* Sequence tagging (e.g. POS/NER): UDPOS, CoNLL2000Chunking
* Question answering: SQuAD1, SQuAD2 
* Text classification: AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, YelpReviewFull, YahooAnswers, AmazonReviewPolarity, AmazonReviewFull, IMDB

For example, to access the raw text from the AG_NEWS dataset:

  .. code-block:: python

      >>> from torchtext.datasets import AG_NEWS
      >>> train_iter = AG_NEWS(split='train')
      >>> next(train_iter)
      >>> # Or iterate with for loop
      >>> for (label, line) in train_iter:
      >>>     print(label, line)
      >>> # Or send to DataLoader
      >>> from torch.utils.data import DataLoader
      >>> train_iter = AG_NEWS(split='train')
      >>> dataloader = DataLoader(train_iter, batch_size=8, shuffle=False)

A tutorial for the end-to-end text classification workflow can be founc in `TEXT CLASSIFICATION WITH TORCHTEXT <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html>`_

[Prototype] Experimental Code
=============================

We have re-written several building blocks under ``torchtext.experimental``:

* Transforms
* Vocabulary
* Vectors

These prototype building blocks in the experimental folder are available in the nightly release only. The nightly packages are accessible via Pip and Conda for Windows, Mac, and Linux. For example, Linux users can install the nightly wheels with the following command::

    pip install --pre torch torchtext -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html  

For more detailed instructions, please refer to `Install PyTorch <https://pytorch.org/get-started/locally/>`_. It should be noted that the new building blocks are still under development, and the APIs have not been solidified.

[BC Breaking] Legacy
====================

In v0.9.0 release, we move the following legacy code to `torchtext.legacy <#legacy>`_. This is part of the work to modernize the torchtext library and the motivation has been discussed in `Issue #664 <https://github.com/pytorch/text/issues/664>`_:

* torchtext.legacy.data.field
* torchtext.legacy.data.batch
* torchtext.legacy.data.example
* torchtext.legacy.data.iterator
* torchtext.legacy.data.pipeline
* torchtext.legacy.datasets

We have a `migration tutorial <https://fburl.com/9hbq843y>`_ to help users switch to the torchtext datasets in ``v0.9.0`` release. For the users who still want the legacy components, they can add ``legacy`` to the import path.  

Disclaimer on Datasets
======================

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
