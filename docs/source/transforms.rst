.. role:: hidden
    :class: hidden-section

torchtext.transforms
===========================

.. automodule:: torchtext.transforms
.. currentmodule:: torchtext.transforms

Transforms are common text transforms. They can be chained together using :class:`torch.nn.Sequential` or using :class:`torchtext.transforms.Sequential` to support torch-scriptability.

SentencePieceTokenizer
----------------------

.. autoclass:: SentencePieceTokenizer

   .. automethod:: forward

GPT2BPETokenizer
----------------------

.. autoclass:: GPT2BPETokenizer

   .. automethod:: forward

CLIPTokenizer
----------------------

.. autoclass:: CLIPTokenizer

   .. automethod:: forward

VocabTransform
--------------

.. autoclass:: VocabTransform

   .. automethod:: forward

ToTensor
--------

.. autoclass:: ToTensor

   .. automethod:: forward

LabelToIndex
------------

.. autoclass:: LabelToIndex

   .. automethod:: forward

Truncate
--------

.. autoclass:: Truncate

   .. automethod:: forward

AddToken
--------

.. autoclass:: AddToken

   .. automethod:: forward

Sequential
----------

.. autoclass:: Sequential

   .. automethod:: forward
