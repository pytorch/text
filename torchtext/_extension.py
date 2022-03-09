def _init_extension():
    import importlib
    import os

    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_torchtext")
    if ext_specs is None:
        raise ImportError("torchtext C++ Extension is not found.")
    torch.ops.load_library(ext_specs.origin)
    torch.classes.load_library(ext_specs.origin)
