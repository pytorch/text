#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import torch
import torchtext
from parameterized import parameterized
from ..common.torchtext_test_case import TorchtextTestCase
from ..common.cache_utils import check_cache_status


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
