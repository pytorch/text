import hashlib
import json

from torchtext.experimental.datasets import sst2

from ..common.case_utils import skipIfNoModule
from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    @skipIfNoModule("torchdata")
    def test_sst2_dataset(self):
        split = ("train", "dev", "test")
        train_dp, dev_dp, test_dp = sst2.SST2(split=split)

        # verify hashes of first line in dataset
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(train_dp)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["train"],
        )
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(dev_dp)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["dev"],
        )
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(test_dp)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["test"],
        )
