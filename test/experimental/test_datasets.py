import hashlib
import json

from torchtext.experimental.datasets import sst2

from ..common.assets import _ASSET_DIR
from ..common.case_utils import skipIfNoModule
from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    @skipIfNoModule("torchdata")
    def test_sst2__dataset(self):

        split = ("train", "dev", "test")
        train_dataset, dev_dataset, test_dataset = sst2.SST2(
            split=split, root=_ASSET_DIR, validate_hash=False
        )

        # verify datasets objects are instances of SST2Dataset
        for dataset in (train_dataset, dev_dataset, test_dataset):
            self.assertTrue(isinstance(dataset, sst2.SST2Dataset))

        # verify hashes of first line in dataset
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(train_dataset)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["train"],
        )
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(dev_dataset)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["dev"],
        )
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(test_dataset)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["test"],
        )
