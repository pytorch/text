import os
import json
import torchtext
from .parameterized_utils import load_params

CACHE_STATUS_FILE = '.data/cache_status_file.json'


def check_cache_status():
    assert os.path.exists(CACHE_STATUS_FILE), "Cache status file does not exists"

def generate_data_cache():
    # cache already created, nothing to do
    if os.path.exists(CACHE_STATUS_FILE):
        return

    raw_data_info = load_params('raw_datasets.jsonl')
    cache_status = {}
    for info in raw_data_info:
        info = info.args[0]
        dataset_name = info['dataset_name']
        split = info['split']
        if dataset_name not in cache_status:
            cache_status[dataset_name] = {}
        try:
            if dataset_name == 'WMT14':
                pass
            else:
                _ = torchtext.datasets.DATASETS[dataset_name](split=split)
            cache_status[dataset_name][split] = {'status': 'success', 'reason': 'No exception thrown'}
        except Exception as e:
            cache_status[dataset_name][split] = {'status': 'fail', 'reason': str(e)}

    with open(CACHE_STATUS_FILE, 'w') as f:
        json.dump(cache_status, f)


if __name__ == "__main__":
    generate_data_cache()
