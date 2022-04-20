import os

from ._extension import _init_extension

_init_extension()
del _init_extension

# the following import has to happen first in order to load the torchtext C++ library
from torchtext import _extension  # noqa: F401

_TEXT_BUCKET = "https://download.pytorch.org/models/text/"
_CACHE_DIR = os.path.expanduser("~/.torchtext/cache")

from . import data, datasets, experimental, functional, models, nn, transforms, utils, vocab

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = ["data", "nn", "datasets", "utils", "vocab", "transforms", "functional", "models", "experimental"]
