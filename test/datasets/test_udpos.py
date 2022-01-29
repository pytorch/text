import os
import random
import string
import zipfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.udpos import UDPOS

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "UDPOS")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ["train.txt", "dev.txt", "test.txt"]:
        txt_file = os.path.join(temp_dataset_dir, file_name)
        mocked_lines = mocked_data[os.path.splitext(file_name)[0]]
        with open(txt_file, "w") as f:
            for i in range(5):
                label = seed % 2
                rand_strings = [random.choice(string.ascii_letters) for i in range(seed)]
                rand_label_1 = [random.choice(string.ascii_letters) for i in range(seed)]
                rand_label_2 = [random.choice(string.ascii_letters) for i in range(seed)]
                # one token per line (each sample ends with an extra \n)
                for rand_string, label_1, label_2 in zip(rand_strings, rand_label_1, rand_label_2):
                    f.write(f"{rand_string}\t{label_1}\t{label_2}\n")
                f.write("\n")
                dataset_line = (rand_strings, rand_label_1, rand_label_2)
                # append line to correct dataset split
                mocked_lines.append(dataset_line)
                seed += 1

    # en-ud-v2.zip
    compressed_dataset_path = os.path.join(base_dir, "en-ud-v2.zip")
    # create zip file from dataset folder
    with zipfile.ZipFile(compressed_dataset_path, "w") as zip_file:
        for file_name in ("train.txt", "dev.txt", "test.txt"):
            txt_file = os.path.join(temp_dataset_dir, file_name)
            zip_file.write(txt_file, arcname=os.path.join("UDPOS", file_name))

    return mocked_data


class TestUDPOS(TempDirMixin, TorchtextTestCase):
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

    @parameterized.expand(["train", "valid", "test"])
    def test_udpos(self, split):
        dataset = UDPOS(root=self.root_dir, split=split)
        samples = list(dataset)
        expected_samples = self.samples[split] if split != "valid" else self.samples["dev"]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "valid", "test"])
    def test_udpos_split_argument(self, split):
        dataset1 = UDPOS(root=self.root_dir, split=split)
        (dataset2,) = UDPOS(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
