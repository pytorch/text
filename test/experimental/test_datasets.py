from torchtext.experimental.datasets import sst2

from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    def test_sst2_dataset(self):

        split = ("train", "dev", "test")
        train_dp, dev_dp, test_dp = sst2.SST2(split=split)

        self.assertEqual(len(list(train_dp)), sst2.NUM_LINES["train"])
        self.assertEqual(len(list(dev_dp)), sst2.NUM_LINES["dev"])
        self.assertEqual(len(list(test_dp)), sst2.NUM_LINES["test"])
