import os
import zipfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.mnli import MNLI

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "MNLI")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ["multinli_1.0_train.txt", "multinli_1.0_dev_matched.txt", "multinli_1.0_dev_mismatched.txt"]:
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(
                "gold_label\tsentence1_binary_parse\tsentence2_binary_parse\tsentence1_parse\tsentence2_parse\tsentence1\tsentence2\tpromptID\tpairID\tgenre\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5"
            )
            for i in range(5):
                label = seed % 3
                rand_string = get_random_unicode(seed)
                dataset_line = (label, rand_string, rand_string)
                f.write(
                    f"{label}\t{rand_string}\t{rand_string}\t{rand_string}\t{rand_string}\t{rand_string}\t{rand_string}\t{i}\t{i}\t{i}\t{i}\t{i}\t{i}\t{i}\t{i}\n"
                )

                # append line to correct dataset split
                mocked_data[os.path.splitext(file_name)[0]].append(dataset_line)
                seed += 1

    compressed_dataset_path = os.path.join(base_dir, "multinli_1.0.zip")
    # create zip file from dataset folder
    with zipfile.ZipFile(compressed_dataset_path, "w") as zip_file:
        for file_name in ("multinli_1.0_train.txt", "multinli_1.0_dev_matched.txt", "multinli_1.0_dev_mismatched.txt"):
            txt_file = os.path.join(temp_dataset_dir, file_name)
            zip_file.write(txt_file, arcname=os.path.join("multinli_1.0", file_name))

    return mocked_data


class TestMNLI(TempDirMixin, TorchtextTestCase):
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

    @parameterized.expand(["train", "dev_matched", "dev_mismatched"])
    def test_mnli(self, split):
        dataset = MNLI(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "dev_matched", "dev_mismatched"])
    def test_sst2_split_argument(self, split):
        dataset1 = MNLI(root=self.root_dir, split=split)
        (dataset2,) = MNLI(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
