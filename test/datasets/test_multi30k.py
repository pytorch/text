import os
import random
import string
import tarfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.amazonreviewpolarity import Multi30k

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "Multi30k")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.de", "train.en", "val.de", "val.en", "test.de", "test.en"):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w") as f:
            for i in range(5):
                rand_string = " ".join(
                    random.choice(string.ascii_letters) for i in range(seed)
                )
                dataset_line = (label, f"{rand_string} {rand_string}")
                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                f.write(f'"{label}","{rand_string}","{rand_string}"\n')
                seed += 1

    compressed_dataset_path = os.path.join(
        base_dir, "amazon_review_polarity_csv.tar.gz"
    )
    # create tar file from dataset folder
    with tarfile.open(compressed_dataset_path, "w:gz") as tar:
        tar.add(temp_dataset_dir, arcname="amazon_review_polarity_csv")

    return mocked_data


class TestAmazonReviewPolarity(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

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
    def test_multi30k(self, split):
        dataset = AmazonReviewPolarity(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_multi30k_split_argument(self, split):
        dataset1 = Multi30k(root=self.root_dir, split=split)
        (dataset2,) = Multi30k(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
