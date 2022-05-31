import os

# the following import has to happen first in order to load the torchtext C++ library
from torchtext import _extension  # noqa: F401

_TEXT_BUCKET = "https://download.pytorch.org/models/text/"

_TORCH_HOME = os.getenv("TORCH_HOME")
if _TORCH_HOME is None:
    _TORCH_HOME = "~/.cache/torch"  # default
_CACHE_DIR = os.path.expanduser(os.path.join(_TORCH_HOME, "text"))

from . import data, datasets, experimental, functional, models, nn, transforms, utils, vocab

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = ["data", "nn", "datasets", "utils", "vocab", "transforms", "functional", "models", "experimental"]
