from . import data
from . import nn
from . import datasets
from . import utils
from . import vocab
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
           'experimental',
           'legacy']


_init_extension()


del _init_extension
