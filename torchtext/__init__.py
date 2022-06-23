import os

from torch.hub import _get_torch_home

# the following import has to happen first in order to load the torchtext C++ library
from torchtext import _extension  # noqa: F401

_TEXT_BUCKET = "https://download.pytorch.org/models/text/"

_CACHE_DIR = os.path.expanduser(os.path.join(_get_torch_home(), "text"))

from . import data, datasets, prototype, functional, models, nn, transforms, utils, vocab, experimental

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = [
    "data",
    "nn",
    "datasets",
    "utils",
    "vocab",
    "transforms",
    "functional",
    "models",
    "prototype",
    "experimental",
]
