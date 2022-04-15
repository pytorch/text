import os
import tarfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.imdb import IMDB

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "IMDB")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for split in ("train", "test"):
        neg_dir = os.path.join(temp_dataset_dir, split, "neg")
        pos_dir = os.path.join(temp_dataset_dir, split, "pos")
        os.makedirs(neg_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)

        for i in range(5):
            # all negative labels are read first before positive labels in the
            # IMDB dataset implementation
            label = "neg" if i < 2 else "pos"
            cur_dir = pos_dir if label == "pos" else neg_dir
            txt_file = os.path.join(cur_dir, f"{i}{i}_{i}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                rand_string = get_random_unicode(seed)
                dataset_line = (label, rand_string)
                # append line to correct dataset split
                mocked_data[split].append(dataset_line)
                f.write(rand_string)
                seed += 1

    compressed_dataset_path = os.path.join(base_dir, "aclImdb_v1.tar.gz")
    # create tar file from dataset folder
    with tarfile.open(compressed_dataset_path, "w:gz") as tar:
        tar.add(temp_dataset_dir, arcname="aclImdb_v1")

    return mocked_data


class TestIMDB(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = _get_mock_dataset(cls.root_dir)
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(["train", "test"])
    def test_imdb(self, split):
        dataset = IMDB(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_imdb_split_argument(self, split):
        dataset1 = IMDB(root=self.root_dir, split=split)
        (dataset2,) = IMDB(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
