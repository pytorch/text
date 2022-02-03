#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import hashlib
import json
import unittest

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
    @unittest.skip("Skipping test due to invalid URL. Enable it back once WMT14 is fixed")
    def test_raw_text_name_property(self, info):
        dataset_name = info["dataset_name"]
        split = info["split"]

        if dataset_name == "WMT14":
            data_iter = torchtext.experimental.datasets.raw.DATASETS[dataset_name](split=split)
        else:
            return
        self.assertEqual(str(data_iter), dataset_name)

    @parameterized.expand(load_params("raw_datasets.jsonl"), name_func=_raw_text_custom_name_func)
    @unittest.skip("Skipping test due to invalid URL. Enable it back once WMT14 is fixed")
    def test_raw_text_classification(self, info):
        dataset_name = info["dataset_name"]
        split = info["split"]

        if dataset_name == "WMT14":
            data_iter = torchtext.experimental.datasets.raw.DATASETS[dataset_name](split=split)
            self.assertEqual(len(data_iter), info["NUM_LINES"])
            self.assertEqual(
                hashlib.md5(json.dumps(next(data_iter), sort_keys=True).encode("utf-8")).hexdigest(), info["first_line"]
            )
            self.assertEqual(torchtext.experimental.datasets.raw.URLS[dataset_name], info["URL"])
            self.assertEqual(torchtext.experimental.datasets.raw.MD5[dataset_name], info["MD5"])
        else:
            return
        del data_iter
