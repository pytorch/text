from pathlib import Path
import glob
import shutil
import os

_ASSET_DIR = (Path(__file__).parent.parent / "asset").resolve()


def get_asset_path(*path_components):
    """Get the path to the file under `test/assets` directory."""
    return str(_ASSET_DIR.joinpath(*path_components))


def conditional_remove(f):
    for path in glob.glob(f):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
