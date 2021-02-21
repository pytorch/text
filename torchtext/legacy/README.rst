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
