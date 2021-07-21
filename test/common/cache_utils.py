import os
import json
import torchtext
from .parameterized_utils import load_params

CACHE_STATUS_FILE = os.path.join(os.path.expanduser('~/.torchtext/cache'), 'cache_status_file.json')


def check_cache_status():
    assert os.path.exists(CACHE_STATUS_FILE), "Cache status file does not exists"
    with open(CACHE_STATUS_FILE, 'r') as f:
        missing_datasets = []
        cache_status = json.load(f)
        for dataset_name in cache_status:
            for split in cache_status[dataset_name]:
                if cache_status[dataset_name][split]['status'] == "fail":
                    missing_datasets.append(dataset_name + '_' + split)
        if missing_datasets:
            raise FileNotFoundError("Failing all raw dataset unit tests as cache is missing {} datasets".format(missing_datasets))


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
