import os
import random
import string
import tarfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.yelpreviewpolarity import YelpReviewPolarity
from torchtext.datasets.yelpreviewfull import YelpReviewFull

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir, dataset_name):
    """
    root_dir: directory to the mocked dataset
    dataset_name: YelpReviewPolarity or YelpReviewFull
    """
    assert dataset_name in ("YelpReviewPolarity", "YelpReviewFull")
    base_dir = os.path.join(root_dir, dataset_name)
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.csv", "test.csv"):
        csv_file = os.path.join(temp_dataset_dir, file_name)
        mocked_lines = mocked_data[os.path.splitext(file_name)[0]]
        with open(csv_file, "w") as f:
            for i in range(5):
                label = seed % 2 + 1 if dataset_name == "YelpReviewPolarity" else seed % 5 + 1
                rand_string = " ".join(
                    random.choice(string.ascii_letters) for i in range(seed)
                )
                dataset_line = (label, f"{rand_string}")
                f.write(f'"{label}","{rand_string}"\n')

                # append line to correct dataset split
                mocked_lines.append(dataset_line)
                seed += 1

    compressed_file = f"yelp_review_{'polarity' if dataset_name == 'YelpReviewPolarity' else 'full'}_csv"
    print(compressed_file)
    compressed_dataset_path = os.path.join(base_dir, compressed_file + ".tar.gz")
    # create gz file from dataset folder
    with tarfile.open(compressed_dataset_path, "w:gz") as tar:
        tar.add(temp_dataset_dir, arcname=compressed_file)

    return mocked_data


class TestYelpReviewPolarity(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = _get_mock_dataset(cls.root_dir, dataset_name="YelpReviewPolarity")
        cls.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True
        )
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(["train", "test"])
    def test_yelpreviewpolarity(self, split):
        dataset = YelpReviewPolarity(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_yelpreviewpolarity_split_argument(self, split):
        dataset1 = YelpReviewPolarity(root=self.root_dir, split=split)
        (dataset2,) = YelpReviewPolarity(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)


class TestYelpReviewFull(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = _get_mock_dataset(cls.root_dir, dataset_name="YelpReviewFull")
        cls.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True
        )
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(["train", "test"])
    def test_yelpreviewfull(self, split):
        dataset = YelpReviewFull(root=self.root_dir, split=split)
        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_yelpreviewfull_split_argument(self, split):
        dataset1 = YelpReviewFull(root=self.root_dir, split=split)
        (dataset2,) = YelpReviewFull(root=self.root_dir, split=(split,))
        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
