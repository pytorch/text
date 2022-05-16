import os
import random
import tarfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.stsb import STSB

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "STSB")
    temp_dataset_dir = os.path.join(base_dir, "stsbenchmark")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name, name in zip(["sts-train.csv", "sts-dev.csv" "sts-test.csv"], ["train", "dev", "test"]):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            for i in range(5):
                label = random.uniform(0, 5)
                rand_string_1 = get_random_unicode(seed)
                rand_string_2 = get_random_unicode(seed + 1)
                rand_string_3 = get_random_unicode(seed + 2)
                rand_string_4 = get_random_unicode(seed + 3)
                rand_string_5 = get_random_unicode(seed + 4)
                dataset_line = (i, label, rand_string_4, rand_string_5)
                # append line to correct dataset split
                mocked_data[name].append(dataset_line)
                f.write(
                    f"{rand_string_1}\t{rand_string_2}\t{rand_string_3}\t{i}\t{label}\t{rand_string_4}\t{rand_string_5}\n"
                )
                seed += 1
            # case with quotes to test arg `quoting=csv.QUOTE_NONE`
            dataset_line = (i, label, rand_string_4, rand_string_5)
            # append line to correct dataset split
            mocked_data[name].append(dataset_line)
            f.write(
                f'{rand_string_1}"\t"{rand_string_2}\t{rand_string_3}\t{i}\t{label}\t{rand_string_4}\t{rand_string_5}\n'
            )


    compressed_dataset_path = os.path.join(base_dir, "Stsbenchmark.tar.gz")
    # create tar file from dataset folder
    with tarfile.open(compressed_dataset_path, "w:gz") as tar:
        tar.add(temp_dataset_dir, arcname="stsbenchmark")

    return mocked_data


class TestSTSB(TempDirMixin, TorchtextTestCase):
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

    @parameterized.expand(["train", "dev", "test"])
    def test_stsb(self, split):
        dataset = STSB(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "dev", "test"])
    def test_stsb_split_argument(self, split):
        dataset1 = STSB(root=self.root_dir, split=split)
        (dataset2,) = STSB(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
