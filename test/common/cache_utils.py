import os
import json
import torchtext
import functools
import unittest
from .parameterized_utils import load_params

CACHE_STATUS_FILE = '.data/cache_status_file.json'


def check_cache_status(_func=None, *, input_dataset_name=None, input_split=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert os.path.exists(CACHE_STATUS_FILE), "Cache status file does not exists"
            if input_dataset_name is None:
                if isinstance(args[1], str):
                    dataset_name = args[1]
                elif isinstance(args[1], dict):
                    dataset_name = args[1]['dataset_name']
                else:
                    raise Exception('cannot find dataset_name in arguments')
            else:
                dataset_name = input_dataset_name

            if input_split is None:
                if isinstance(args[1], dict):
                    split = args[1]['split']
                elif len(args) > 2 and isinstance(args[2], str):
                    split = args[2]
                else:
                    raise Exception('cannot find split in arguments')
            else:
                split = input_split

            # failure modes
            with open(CACHE_STATUS_FILE, 'r') as f:
                cache_status = json.load(f)
                if dataset_name not in cache_status:
                    raise KeyError('{} not found in cache status file.'.format(dataset_name))

                if split not in cache_status[dataset_name]:
                    raise KeyError('{} split not found in cache status file for {}'.format(split, dataset_name))

                if cache_status[dataset_name][split]['status'] == 'fail':
                    raise unittest.SkipTest('Data not cached due to "{}"'.format(cache_status[dataset_name][split]['reason']))

            func(*args, **kwargs)
        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def generate_data_cache():
    # cache already created, nothing to do
    if os.path.exists(CACHE_STATUS_FILE):
        return

    raw_data_info = load_params('raw_datasets.json')
    cache_status = {}
    for info in raw_data_info:
        info = info.args[0]
        dataset_name = info['dataset_name']
        split = info['split']
        if dataset_name not in cache_status:
            cache_status[dataset_name] = {}
        try:
            if dataset_name == "Multi30k" or dataset_name == 'WMT14':
                _ = torchtext.experimental.datasets.raw.DATASETS[dataset_name](split=split)
            else:
                _ = torchtext.datasets.DATASETS[dataset_name](split=split)
            cache_status[dataset_name][split] = {'status': 'success', 'reason': 'No exception thrown'}
        except Exception as e:
            cache_status[dataset_name][split] = {'status': 'fail', 'reason': str(e)}

    with open(CACHE_STATUS_FILE, 'w') as f:
        json.dump(cache_status, f)


if __name__ == "__main__":
    generate_data_cache()
