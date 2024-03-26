import os

from torch.hub import _get_torch_home

_WARN = True
_TORCHTEXT_DEPRECATION_MSG = (
    "\n/!\ IMPORTANT WARNING ABOUT TORCHTEXT STATUS /!\ \n"
    "Torchtext is deprecated and the last released version will be 0.18 (this one). "
    "You can silence this warning by calling the following at the beginnign of your scripts: "
    "`import torchtext; torchtext.disable_torchtext_deprecation_warning()`"
)

def disable_torchtext_deprecation_warning():
    global _WARN
    _WARN = False

# the following import has to happen first in order to load the torchtext C++ library
from torchtext import _extension  # noqa: F401

_TEXT_BUCKET = "https://download.pytorch.org/models/text/"

_CACHE_DIR = os.path.expanduser(os.path.join(_get_torch_home(), "text"))

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
