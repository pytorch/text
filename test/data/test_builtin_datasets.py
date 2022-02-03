#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import hashlib
import json

import torchtext
from parameterized import parameterized

from ..common.cache_utils import check_cache_status
from ..common.parameterized_utils import load_params
from ..common.torchtext_test_case import TorchtextTestCase


def _raw_text_custom_name_func(testcase_func, param_num, param):
    info = param.args[0]
    name_info = [info["dataset_name"], info["split"]]
    return "%s_%s" % (testcase_func.__name__, parameterized.to_safe_name("_".join(name_info)))


class TestDataset(TorchtextTestCase):
    @classmethod
    def setUpClass(cls):
        check_cache_status()

    @parameterized.expand(load_params("raw_datasets.jsonl"), name_func=_raw_text_custom_name_func)
    def test_raw_text_classification(self, info):
        dataset_name = info["dataset_name"]
        split = info["split"]

        if dataset_name == "WMT14":
            return
        else:
            data_iter = torchtext.datasets.DATASETS[dataset_name](split=split)
        self.assertEqual(
            hashlib.md5(json.dumps(next(iter(data_iter)), sort_keys=True).encode("utf-8")).hexdigest(),
            info["first_line"],
        )
        if dataset_name == "AG_NEWS":
            self.assertEqual(torchtext.datasets.URLS[dataset_name][split], info["URL"])
            self.assertEqual(torchtext.datasets.MD5[dataset_name][split], info["MD5"])
        elif dataset_name == "WMT14":
            return
        else:
            self.assertEqual(torchtext.datasets.URLS[dataset_name], info["URL"])
            self.assertEqual(torchtext.datasets.MD5[dataset_name], info["MD5"])
        del data_iter

    @parameterized.expand(list(sorted(torchtext.datasets.DATASETS.keys())))
    def test_raw_datasets_split_argument(self, dataset_name):
        if "statmt" in torchtext.datasets.URLS[dataset_name]:
            return
        dataset = torchtext.datasets.DATASETS[dataset_name]
        train1 = dataset(split="train")
        (train2,) = dataset(split=("train",))
        for d1, d2 in zip(train1, train2):
            self.assertEqual(d1, d2)
            # This test only aims to exercise the argument parsing and uses
            # the first line as a litmus test for correctness.
            break
        # Exercise default constructor
        _ = dataset()
