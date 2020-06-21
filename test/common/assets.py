from pathlib import Path

_ASSET_DIR = (Path(__file__).parent.parent / "asset").resolve()


def get_asset_path(*path_components):
    """Get the path to the file under `test/assets` directory."""
    return str(_ASSET_DIR.joinpath(*path_components))
