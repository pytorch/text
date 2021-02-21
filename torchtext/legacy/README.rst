Legacy
======

In v0.9.0 release, we move the following legacy code to `torchtext.legacy <#legacy>`_. This is part of the work to revamp the torchtext library and the motivation has been discussed in `Issue #664 <https://github.com/pytorch/text/issues/664>`_:

* ``torchtext.legacy.data.field``
* ``torchtext.legacy.data.batch``
* ``torchtext.legacy.data.example``
* ``torchtext.legacy.data.iterator``
* ``torchtext.legacy.data.pipeline``
* ``torchtext.legacy.datasets``

We have a `migration tutorial <https://fburl.com/9hbq843y>`_ to help users switch to the torchtext datasets in ``v0.9.0`` release. For the users who still want the legacy components, they can add ``legacy`` to the import path.

Another option is to import ``torchtext.legacy`` as ``torchtext``. For example:

With `torchtext v0.8.1`

  .. code-block:: python

      >>> import torchtext
      >>> import torch

      >>> TEXT = torchtext.data.Field(tokenize=torchtext.data.get_tokenizer('basic_english'),
                                      init_token='<SOS>', eos_token='<EOS>', lower=True)
      >>> LABEL = torchtext.data.LabelField(dtype = torch.long)
      >>> train_split, test_split = torchtext.datasets.IMDB.splits(TEXT, LABEL)
      >>> TEXT.build_vocab(train_split)
      >>> LABEL.build_vocab(train_split)

      >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      >>> train_iterator, test_iterator = torchtext.data.Iterator.splits(
                   (train_split, test_split), batch_size=8, device = device)
      >>> next(iter(train_iterator))

With `torchtext v0.9.0`

  .. code-block:: python

      >>> import torchtext.legacy as torchtext  # need to change only one line
      >>> import torch

      >>> TEXT = torchtext.data.Field(tokenize=torchtext.data.get_tokenizer('basic_english'),
                                      init_token='<SOS>', eos_token='<EOS>', lower=True)
      >>> LABEL = torchtext.data.LabelField(dtype = torch.long)
      >>> train_split, test_split = torchtext.datasets.IMDB.splits(TEXT, LABEL)
      >>> TEXT.build_vocab(train_split)
      >>> LABEL.build_vocab(train_split)

      >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      >>> train_iterator, test_iterator = torchtext.data.Iterator.splits(
                   (train_split, test_split), batch_size=8, device = device)
      >>> next(iter(train_iterator))
