#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import torch
import torchtext
import json
import hashlib
from parameterized import parameterized
from ..common.torchtext_test_case import TorchtextTestCase
from ..common.parameterized_utils import load_params
from ..common.cache_utils import check_cache_status


def _raw_text_custom_name_func(testcase_func, param_num, param):
    info = param.args[0]
    name_info = [info['dataset_name'], info['split']]
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(name_info))
    )


class TestDataset(TorchtextTestCase):
    @classmethod
    def setUpClass(cls):
        check_cache_status()

    def _helper_test_func(self, length, target_length, results, target_results):
        self.assertEqual(length, target_length)
        if isinstance(target_results, list):
            target_results = torch.tensor(target_results, dtype=torch.int64)
        if isinstance(target_results, tuple):
            target_results = tuple(torch.tensor(item, dtype=torch.int64) for item in target_results)
        self.assertEqual(results, target_results)

    def test_raw_ag_news(self):
        train_iter, test_iter = torchtext.datasets.AG_NEWS()
        self._helper_test_func(len(train_iter), 120000, next(train_iter)[1][:25], 'Wall St. Bears Claw Back ')
        self._helper_test_func(len(test_iter), 7600, next(test_iter)[1][:25], 'Fears for T N pension aft')
        del train_iter, test_iter

    @parameterized.expand(
        load_params('raw_datasets.jsonl'),
        name_func=_raw_text_custom_name_func)
    def test_raw_text_name_property(self, info):
        dataset_name = info['dataset_name']
        split = info['split']

        if dataset_name == 'WMT14':
            data_iter = torchtext.experimental.datasets.raw.DATASETS[dataset_name](split=split)
        else:
            data_iter = torchtext.datasets.DATASETS[dataset_name](split=split)

        self.assertEqual(str(data_iter), dataset_name)

    @parameterized.expand(
        load_params('raw_datasets.jsonl'),
        name_func=_raw_text_custom_name_func)
    def test_raw_text_classification(self, info):
        dataset_name = info['dataset_name']
        split = info['split']

        if dataset_name == 'WMT14':
            data_iter = torchtext.experimental.datasets.raw.DATASETS[dataset_name](split=split)
        else:
            data_iter = torchtext.datasets.DATASETS[dataset_name](split=split)
        self.assertEqual(len(data_iter), info['NUM_LINES'])
        self.assertEqual(hashlib.md5(json.dumps(next(data_iter), sort_keys=True).encode('utf-8')).hexdigest(), info['first_line'])
        if dataset_name == "AG_NEWS":
            self.assertEqual(torchtext.datasets.URLS[dataset_name][split], info['URL'])
            self.assertEqual(torchtext.datasets.MD5[dataset_name][split], info['MD5'])
        elif dataset_name == "WMT14":
            self.assertEqual(torchtext.experimental.datasets.raw.URLS[dataset_name], info['URL'])
            self.assertEqual(torchtext.experimental.datasets.raw.MD5[dataset_name], info['MD5'])
        else:
            self.assertEqual(torchtext.datasets.URLS[dataset_name], info['URL'])
            self.assertEqual(torchtext.datasets.MD5[dataset_name], info['MD5'])
        del data_iter

    @parameterized.expand(
        load_params('raw_datasets.jsonl'),
        name_func=_raw_text_custom_name_func)
    def test_dataset_iterator_offsets(self, info):
        dataset_name = info['dataset_name']

        if dataset_name != "AG_NEWS":
            return

        split = info['split']
        offset = info['offset']

        if dataset_name == 'WMT14':
            data_iter = torchtext.experimental.datasets.raw.DATASETS[dataset_name](split=split, offset=offset)
        else:
            data_iter = torchtext.datasets.DATASETS[dataset_name](split=split, offset=offset)
        self.assertEqual(hashlib.md5(json.dumps(next(data_iter), sort_keys=True).encode('utf-8')).hexdigest(), info['offset_line'])

    @parameterized.expand(list(sorted(torchtext.datasets.DATASETS.keys())))
    def test_raw_datasets_split_argument(self, dataset_name):
        if 'statmt' in torchtext.datasets.URLS[dataset_name]:
            return
        dataset = torchtext.datasets.DATASETS[dataset_name]
        train1 = dataset(split='train')
        train2, = dataset(split=('train',))
        for d1, d2 in zip(train1, train2):
            self.assertEqual(d1, d2)
            # This test only aims to exercise the argument parsing and uses
            # the first line as a litmus test for correctness.
            break
        # Exercise default constructor
        _ = dataset()

    def test_next_method_dataset(self):
        train_iter, test_iter = torchtext.datasets.AG_NEWS()
        for_count = 0
        next_count = 0
        for line in train_iter:
            for_count += 1
            try:
                next(train_iter)
                next_count += 1
            except:
                break
        self.assertEqual((for_count, next_count), (60000, 60000))
