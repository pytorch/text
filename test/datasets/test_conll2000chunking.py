import gzip
import os
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.conll2000chunking import CoNLL2000Chunking

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "CoNLL2000Chunking")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.txt", "test.txt"):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        mocked_lines = mocked_data[os.path.splitext(file_name)[0]]
        with open(txt_file, "w", encoding="utf-8") as f:
            for i in range(5):
                rand_strings = [get_random_unicode(seed)]
                rand_label_1 = [get_random_unicode(seed)]
                rand_label_2 = [get_random_unicode(seed)]
                # one token per line (each sample ends with an extra \n)
                for rand_string, label_1, label_2 in zip(rand_strings, rand_label_1, rand_label_2):
                    f.write(f"{rand_string} {label_1} {label_2}\n")
                f.write("\n")
                dataset_line = (rand_strings, rand_label_1, rand_label_2)
                # append line to correct dataset split
                mocked_lines.append(dataset_line)
                seed += 1

        # create gz file from dataset folder
        compressed_dataset_path = os.path.join(base_dir, f"{file_name}.gz")
        with gzip.open(compressed_dataset_path, "wb") as gz_file, open(txt_file, "rb") as file_in:
            gz_file.writelines(file_in)

    return mocked_data


class TestCoNLL2000Chunking(TempDirMixin, TorchtextTestCase):
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
    def test_conll2000chunking(self, split):
        dataset = CoNLL2000Chunking(root=self.root_dir, split=split)
        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_conll2000chunking_split_argument(self, split):
        dataset1 = CoNLL2000Chunking(root=self.root_dir, split=split)
        (dataset2,) = CoNLL2000Chunking(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
