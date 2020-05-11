from . import data
from . import datasets
from . import utils
from . import vocab
from . import experimental

__version__ = '0.6.0'

__all__ = ['data',
           'datasets',
           'utils',
           'vocab',
           'experimental']


def _init_extension():
    import torch
    torch.ops.load_library('torchtext/_torchtext.so')
    torch.classes.load_library('torchtext/_torchtext.so')


_init_extension()


del _init_extension
