.. role:: hidden
    :class: hidden-section

torchtext.transforms
===========================

.. automodule:: torchtext.transforms
.. currentmodule:: torchtext.transforms

Transforms are common text transforms. They can be chained together using :class:`torch.nn.Sequential`

SentencePieceTokenizer
----------------------

.. autoclass:: SentencePieceTokenizer

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
------------

.. autoclass:: Truncate

   .. automethod:: forward
