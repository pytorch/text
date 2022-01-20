import os.path
import random
import string
import tarfile
from collections import defaultdict

from parameterized import parameterized
from torchtext.datasets.amazonreviewpolarity import AmazonReviewPolarity

from ..common.case_utils import TempDirMixin
from ..common.torchtext_test_case import TorchtextTestCase


def get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "AmazonReviewPolarity")
    compressed_dataset_path = os.path.join(
        base_dir, "amazon_review_polarity_csv.tar.gz"
    )
    uncompressed_dataset_dir = os.path.join(base_dir, "amazon_review_polarity_csv")
    os.makedirs(uncompressed_dataset_dir, exist_ok=True)

    # create empty tar file to skip dataset download
    with tarfile.open(compressed_dataset_path, "w:gz") as tar:
        dummy_file_path = os.path.join(base_dir, "dummy_file.txt")
        with open(dummy_file_path, "w") as f:
            pass
        tar.add(dummy_file_path)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.csv", "test.csv"):
        txt_file = os.path.join(uncompressed_dataset_dir, file_name)
        with open(txt_file, "w") as f:
            for i in range(5):
                label = seed % 2 + 1
                rand_string = " ".join(
                    random.choice(string.ascii_letters) for i in range(seed)
                )
                dataset_line = (label, f"{rand_string} {rand_string}")
                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                f.write(f'"{label}","{rand_string}","{rand_string}"\n')
                seed += 1
    return mocked_data


class TestAmazonReviewPolarity(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = get_mock_dataset(cls.root_dir)

    @parameterized.expand(["train", "test"])
    def test_amazon_review_polarity(self, split):
        dataset = AmazonReviewPolarity(
            root=self.root_dir, split=split, validate_hash=False
        )
        n_iter = 0
        for i, (label, text) in enumerate(dataset):
            expected_sample = self.samples[split][i]
            assert label == expected_sample[0]
            assert text == expected_sample[1]
            n_iter += 1
        assert n_iter == len(self.samples[split])

    @parameterized.expand([("train", ("train",)), ("test", ("test",))])
    def test_amazon_review_polarity_split_argument(self, split1, split2):
        dataset1 = AmazonReviewPolarity(
            root=self.root_dir, split=split1, validate_hash=False
        )
        (dataset2,) = AmazonReviewPolarity(
            root=self.root_dir, split=split2, validate_hash=False
        )

        for d1, d2 in zip(dataset1, dataset2):
            self.assertEqual(d1, d2)
