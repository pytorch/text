from . import data
from . import datasets
from . import utils
from . import vocab
from . import experimental


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = ['data',
           'datasets',
           'utils',
           'vocab',
           'experimental']


def _init_extension():
    import os
    import importlib
    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_torchtext")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)
    torch.classes.load_library(ext_specs.origin)


_init_extension()


del _init_extension
