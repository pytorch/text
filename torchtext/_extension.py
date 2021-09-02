import importlib


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _init_extension():
    if is_module_available("torchtext._torchtext"):
        # Note this import has two purposes
        # 1. Make _torchtext accessible by the other modules (regular import)
        # 2. Register torchtext's custom ops bound via TorchScript
        #
        # For 2, normally function calls `torch.ops.load_library` and `torch.classes.load_library`
        # are used. However, in our cases, this is inconvenient and unnecessary.
        #
        # - Why inconvenient?
        # When torchtext is deployed with `pex` format, all the files are deployed as a single zip
        # file, and the extension module is not present as a file with full path. Therefore it is not
        # possible to pass the path to library to `torch.[ops|classes].load_library` functions.
        #
        # - Why unnecessary?
        # When torchtext extension module (C++ module) is available, it is assumed that
        # the extension contains both TorchScript-based binding and PyBind11-based binding.*
        # Under this assumption, simply performing `from torchtext import _torchtext' will load the
        # library which contains TorchScript-based binding as well, and the functions/classes bound
        # via TorchScript become accessible under `torch.ops` and `torch.classes`.
        #
        # *Note that this holds true even when these two bindings are split into two library files and
        # the library that contains PyBind11-based binding (`_torchtext.so` in the following diagram)
        # depends on the other one (`libtorchtext.so`), because when the process tries to load
        # `_torchtext.so` it detects undefined symbols from `libtorchtext.so` and will automatically
        # loads `libtorchtext.so`. (given that the library is found in a search path)
        #
        # [libtorchtext.so] <- [_torchtext.so]
        #
        #
        from torchtext import _torchtext  # noqa
    else:
        raise ImportError("torchtext C++ extension is not available.")
