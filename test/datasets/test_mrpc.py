import os
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.mrpc import MRPC

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "MRPC")
    os.makedirs(base_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name, file_type in [("msr_paraphrase_train.txt", "train"), ("msr_paraphrase_test.txt", "test")]:
        txt_file = os.path.join(base_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
            for i in range(5):
                label = seed % 2
                rand_string_1 = get_random_unicode(seed)
                rand_string_2 = get_random_unicode(seed + 1)
                dataset_line = (label, rand_string_1, rand_string_2)
                f.write(f"{label}\t{i}\t{i}\t{rand_string_1}\t{rand_string_2}\n")

                # append line to correct dataset split
                mocked_data[file_type].append(dataset_line)
                seed += 1

    return mocked_data


class TestMRPC(TempDirMixin, TorchtextTestCase):
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
    def test_mrpc(self, split):
        dataset = MRPC(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "test"])
    def test_sst2_split_argument(self, split):
        dataset1 = MRPC(root=self.root_dir, split=split)
        (dataset2,) = MRPC(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
