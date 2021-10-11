TEXT_BUCKET = 'https://download.pytorch.org/models/text'

from . import data
from . import nn
from . import datasets
from . import utils
from . import vocab
from . import models
from . import experimental
from . import legacy
from ._extension import _init_extension


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = ['data',
           'nn',
           'datasets',
           'utils',
           'vocab',
           'models',
           'experimental',
           'legacy']


_init_extension()


del _init_extension
