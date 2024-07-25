import os
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.penntreebank import PennTreebank

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "PennTreebank")
    os.makedirs(base_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"):
        txt_file = os.path.join(base_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            for i in range(5):
                rand_string = get_random_unicode(seed)
                dataset_line = f"{rand_string}"
                # append line to correct dataset split
                split = file_name.replace("ptb.", "").replace(".txt", "")
                mocked_data[split].append(dataset_line)
                f.write(f"{rand_string}\n")
                seed += 1

    return mocked_data


class TestPennTreebank(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = _get_mock_dataset(os.path.join(cls.root_dir, "datasets"))
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(["train", "valid", "test"])
    def test_penn_treebank_polarity(self, split):
        dataset = PennTreebank(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "valid", "test"])
    def test_penn_treebank_split_argument(self, split):
        dataset1 = PennTreebank(root=self.root_dir, split=split)
        (dataset2,) = PennTreebank(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
