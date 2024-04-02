import warnings
import torchtext
if torchtext._WARN:
    warnings.warn(torchtext._TORCHTEXT_DEPRECATION_MSG)

from . import transforms

__all__ = ["transforms"]
