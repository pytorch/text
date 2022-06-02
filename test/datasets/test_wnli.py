import os
import zipfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.wnli import WNLI

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "WNLI")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.tsv", "test.tsv", "dev.tsv"):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("index\tsentence1\tsentence2\tlabel\n")
            for i in range(5):
                label = seed % 2
                rand_string_1 = get_random_unicode(seed)
                rand_string_2 = get_random_unicode(seed + 1)
                if file_name == "test.tsv":
                    dataset_line = (rand_string_1, rand_string_2)
                    f.write(f"{i}\t{rand_string_1}\t{rand_string_2}\n")
                else:
                    dataset_line = (label, rand_string_1, rand_string_2)
                    f.write(f"{i}\t{rand_string_1}\t{rand_string_2}\t{label}\n")

                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                seed += 1

    compressed_dataset_path = os.path.join(base_dir, "WNLI.zip")
    # create zip file from dataset folder
    with zipfile.ZipFile(compressed_dataset_path, "w") as zip_file:
        for file_name in ("train.tsv", "test.tsv", "dev.tsv"):
            txt_file = os.path.join(temp_dataset_dir, file_name)
            zip_file.write(txt_file, arcname=os.path.join("WNLI", file_name))

    return mocked_data


class TestWNLI(TempDirMixin, TorchtextTestCase):
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

    @parameterized.expand(["train", "test", "dev"])
    def test_wnli(self, split):
        dataset = WNLI(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test", "dev"])
    def test_wnli_split_argument(self, split):
        dataset1 = WNLI(root=self.root_dir, split=split)
        (dataset2,) = WNLI(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
