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
