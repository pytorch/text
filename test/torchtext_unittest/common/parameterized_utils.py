import json
from parameterized import param
import os.path


_TEST_DIR_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..'))


def get_asset_path(*paths):
    """Return full path of a test asset"""
    return os.path.join(_TEST_DIR_PATH, 'asset', *paths)


def load_params(*paths):
    with open(get_asset_path(*paths), 'r') as file:
        return [param(json.loads(line)) for line in file]
