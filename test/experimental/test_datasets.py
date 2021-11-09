import hashlib
import json
import os
import shutil
import tempfile

from torchtext.experimental.datasets import sst2

from ..common.assets import get_asset_path
from ..common.case_utils import skipIfNoModule
from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    @skipIfNoModule("torchdata")
    def test_sst2__dataset(self):
        # copy the asset file into the expected download location
        # note that this is just a zip file with the first 10 lines of the SST2 dataset
        # test if providing a custom hash works with the dummy dataset
        with tempfile.TemporaryDirectory() as dir_name:
            asset_path = get_asset_path(sst2._PATH)
            data_path = os.path.join(dir_name, sst2.DATASET_NAME, sst2._PATH)
            os.makedirs(os.path.join(dir_name, sst2.DATASET_NAME))
            shutil.copy(asset_path, data_path)

            split = ("train", "dev", "test")
            train_dataset, dev_dataset, test_dataset = sst2.SST2(
                split=split, root=dir_name, validate_hash=False
            )

            # verify datasets objects are instances of SST2Dataset
            for dataset in (train_dataset, dev_dataset, test_dataset):
                self.assertTrue(isinstance(dataset, sst2.SST2Dataset))

            # verify hashes of first line in dataset
            self.assertEqual(
                hashlib.md5(
                    json.dumps(next(iter(train_dataset)), sort_keys=True).encode(
                        "utf-8"
                    )
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
