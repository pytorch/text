import os
import shutil
import atexit
import tempfile
from pathlib import Path

_ASSET_DIR = (Path(__file__).parent.parent / "asset").resolve()

_TEMP_DIR = None


def _init_temp_dir():
    """Initialize temporary directory and register clean up at the end of test."""
    global _TEMP_DIR
    _TEMP_DIR = tempfile.TemporaryDirectory()  # noqa
    atexit.register(_TEMP_DIR.cleanup)


def get_asset_path(*path_components, use_temp_dir=False):
    """Get the path to the file under `test/assets` directory.
    When `use_temp_dir` is True, the asset is copied to a temporary location and
    path to the temporary file is returned.
    """
    path = str(_ASSET_DIR.joinpath(*path_components))
    if not use_temp_dir:
        return path

    if _TEMP_DIR is None:
        _init_temp_dir()
    tgt = os.path.join(_TEMP_DIR.name, path_components[-1])
    shutil.copy(path, tgt)
    return tgt
