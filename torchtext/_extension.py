import os
from pathlib import Path

import torch
from torchtext._internal import module_utils as _mod_utils

_LIB_DIR = Path(__file__).parent / "lib"


def _get_lib_path(lib: str):
    suffix = "pyd" if os.name == "nt" else "so"
    path = _LIB_DIR / f"{lib}.{suffix}"
    return path


def _load_lib(lib: str) -> bool:
    """Load extension module

    Note:
        In case `torchtext` is deployed with `pex` format, the library file
        is not in a standard location.
        In this case, we expect that `libtorchtext` is available somewhere
        in the search path of dynamic loading mechanism, so that importing
        `_torchtext` will have library loader find and load `libtorchtext`.
        This is the reason why the function should not raising an error when the library
        file is not found.

    Returns:
        bool:
            True if the library file is found AND the library loaded without failure.
            False if the library file is not found (like in the case where torchtext
            is deployed with pex format, thus the shared library file is
            in a non-standard location.).
            If the library file is found but there is an issue loading the library,
            (such as missing dependency) then this function raises the exception as-is.

    Raises:
        Exception:
            If the library file is found, but there is an issue loading the library file,
            (when underlying `ctype.DLL` throws an exception), this function will pass
            the exception as-is, instead of catching it and returning bool.
            The expected case is `OSError` thrown by `ctype.DLL` when a dynamic dependency
            is not found.
            This behavior was chosen because the expected failure case is not recoverable.
            If a dependency is missing, then users have to install it.
    """
    path = _get_lib_path(lib)
    if not path.exists():
        return False
    torch.ops.load_library(path)
    return True


def _init_extension():
    if not _mod_utils.is_module_available("torchtext._torchtext"):
        raise ImportError("torchtext C++ Extension is not found.")

    _load_lib("libtorchtext")
    # This import is for initializing the methods registered via PyBind11
    # This has to happen after the base library is loaded
    from torchtext import _torchtext  # noqa


_init_extension()
