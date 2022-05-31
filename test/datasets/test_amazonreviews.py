import os
import tarfile
from collections import defaultdict
from unittest.mock import patch

from torchtext.datasets.amazonreviewfull import AmazonReviewFull
from torchtext.datasets.amazonreviewpolarity import AmazonReviewPolarity

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.parameterized_utils import nested_params
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir, base_dir_name):
    """
    root_dir: directory to the mocked dataset
    base_dir_name: AmazonReviewFull or AmazonReviewPolarity
    """
    base_dir = os.path.join(root_dir, base_dir_name)
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.csv", "test.csv"):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            for i in range(5):
                if base_dir_name == AmazonReviewFull.__name__:
                    label = seed % 5 + 1
                else:
                    label = seed % 2 + 1
                label = seed % 2 + 1
                rand_string = get_random_unicode(seed)
                dataset_line = (label, f"{rand_string} {rand_string}")
                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                f.write(f'"{label}","{rand_string}","{rand_string}"\n')
                seed += 1

    if base_dir_name == AmazonReviewFull.__name__:
        archive_file_name = "amazon_review_full_csv"
    else:
        archive_file_name = "amazon_review_polarity_csv"

    compressed_dataset_path = os.path.join(base_dir, f"{archive_file_name}.tar.gz")
    # create tar file from dataset folder
    with tarfile.open(compressed_dataset_path, "w:gz") as tar:
        tar.add(temp_dataset_dir, arcname=archive_file_name)

    return mocked_data


class TestAmazonReviews(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @nested_params([AmazonReviewFull, AmazonReviewPolarity], ["train", "test"])
    def test_amazon_reviews(self, amazon_review_dataset, split):
        expected_samples = _get_mock_dataset(os.path.join(self.root_dir, "datasets"), amazon_review_dataset.__name__)[
            split
        ]
        dataset = amazon_review_dataset(root=self.root_dir, split=split)
        samples = list(dataset)

        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @nested_params([AmazonReviewFull, AmazonReviewPolarity], ["train", "test"])
    def test_amazon_reviews_split_argument(self, amazon_review_dataset, split):
        # call `_get_mock_dataset` to create mock dataset files
        _ = _get_mock_dataset(self.root_dir, amazon_review_dataset.__name__)

        dataset1 = amazon_review_dataset(root=self.root_dir, split=split)
        (dataset2,) = amazon_review_dataset(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
