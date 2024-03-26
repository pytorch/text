import warnings
import torchtext
if torchtext._WARN:
    warnings.warn(torchtext._TORCHTEXT_DEPRECATION_MSG)

from .modules import *  # noqa: F401,F403
