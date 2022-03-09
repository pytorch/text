import os
import zipfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.sst2 import SST2

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "SST2")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name, (col1_name, col2_name) in zip(
        ("train.tsv", "test.tsv", "dev.tsv"),
        ((("sentence", "label"), ("sentence", "label"), ("index", "sentence"))),
    ):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"{col1_name}\t{col2_name}\n")
            for i in range(5):
                label = seed % 2
                rand_string = get_random_unicode(seed)
                if file_name == "test.tsv":
                    dataset_line = (f"{rand_string} .",)
                    f.write(f"{i}\t{rand_string} .\n")
                else:
                    dataset_line = (f"{rand_string} .", label)
                    f.write(f"{rand_string} .\t{label}\n")

                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                seed += 1

    compressed_dataset_path = os.path.join(base_dir, "SST-2.zip")
    # create zip file from dataset folder
    with zipfile.ZipFile(compressed_dataset_path, "w") as zip_file:
        for file_name in ("train.tsv", "test.tsv", "dev.tsv"):
            txt_file = os.path.join(temp_dataset_dir, file_name)
            zip_file.write(txt_file, arcname=os.path.join("SST-2", file_name))

    return mocked_data


class TestSST2(TempDirMixin, TorchtextTestCase):
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

    @parameterized.expand(["train", "test", "dev"])
    def test_sst2(self, split):
        dataset = SST2(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test", "dev"])
    def test_sst2_split_argument(self, split):
        dataset1 = SST2(root=self.root_dir, split=split)
        (dataset2,) = SST2(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
