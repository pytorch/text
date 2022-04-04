import os

from torchtext import _extension  # noqa: F401

_TEXT_BUCKET = "https://download.pytorch.org/models/text/"
_CACHE_DIR = os.path.expanduser("~/.torchtext/cache")

from . import data, datasets, experimental, functional, models, nn, transforms, utils, vocab

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = ["data", "nn", "datasets", "utils", "vocab", "transforms", "functional", "models", "experimental"]
