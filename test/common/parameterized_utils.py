import json

from parameterized import param

from .assets import get_asset_path


def load_params(*paths):
    with open(get_asset_path(*paths), "r") as file:
        return [param(json.loads(line)) for line in file]
