torchtext.experimental.datasets.raw
===================================

.. currentmodule:: torchtext.experimental.datasets.raw

General use cases are as follows: ::


    # import datasets
    from torchtext.experimental.datasets.raw import Multi30k

    train_iter = Multi30k(split='train')

    def tokenize(label, line):
        return line.split()

    tokens_src = []
    tokens_tgt = []

    for line in train_iter:
        src, tgt = line
        tokens_src += tokenize(src)
        tokens_tgt += tokenize(tgt)

The following datasets are available:

.. contents:: Datasets
    :local:


Machine Translation
^^^^^^^^^^^^^^^^^^^

WMT14
~~~~~

.. autofunction:: WMT14


Language Modeling
^^^^^^^^^^^^^^^^^

WMTNewsCrawl
~~~~~~~~~~~~

.. autofunction:: WMTNewsCrawl
