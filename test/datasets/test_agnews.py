import os
import random
import string
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.ag_news import AG_NEWS

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    temp_dataset_dir = os.path.join(root_dir, "AG_NEWS")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.csv", "test.csv"):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w") as f:
            for i in range(5):
                label = seed % 4 + 1
                rand_string = " ".join(
                    random.choice(string.ascii_letters) for i in range(seed)
                )
                dataset_line = (label, f"{rand_string} {rand_string}")
                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                f.write(f'"{label}","{rand_string}","{rand_string}"\n')
                seed += 1

    return mocked_data


class TestAGNews(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []
    patcher = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = _get_mock_dataset(cls.root_dir)
        cls.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True
        )
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(["train", "test"])
    def test_agnews(self, split):
        dataset = AG_NEWS(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            print(sample, expected_sample)
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_agnews_split_argument(self, split):
        dataset1 = AG_NEWS(root=self.root_dir, split=split)
        (dataset2,) = AG_NEWS(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
