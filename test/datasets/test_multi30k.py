import os
import tarfile
from collections import defaultdict
from unittest.mock import patch

from torchtext.datasets import Multi30k

from ..common.parameterized_utils import nested_params
from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "Multi30k")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    for file_name in ("train.de", "train.en", "val.de", "val.en", "test.de", "test.en"):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            for i in range(5):
                rand_string = get_random_unicode(seed)
                f.write(rand_string + "\n")
                mocked_data[file_name].append(rand_string)
                seed += 1

    archive = {}
    archive["train"] = os.path.join(base_dir, "training.tar.gz")
    archive["val"] = os.path.join(base_dir, "validation.tar.gz")
    archive["test"] = os.path.join(base_dir, "mmt16_task1_test.tar.gz")

    for split in ("train", "val", "test"):
        with tarfile.open(archive[split], "w:gz") as tar:
            tar.add(os.path.join(temp_dataset_dir, f"{split}.de"))
            tar.add(os.path.join(temp_dataset_dir, f"{split}.en"))

    return mocked_data


class TestMulti30k(TempDirMixin, TorchtextTestCase):
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

    @nested_params(["train", "valid", "test"], [("de", "en"), ("en", "de")])
    def test_multi30k(self, split, language_pair):
        dataset = Multi30k(root=self.root_dir, split=split, language_pair=language_pair)
        if split == "valid":
            split = "val"
        samples = list(dataset)
        expected_samples = [
            (d1, d2)
            for d1, d2 in zip(
                self.samples[f"{split}.{language_pair[0]}"],
                self.samples[f"{split}.{language_pair[1]}"],
            )
        ]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @nested_params(["train", "valid", "test"], [("de", "en"), ("en", "de")])
    def test_multi30k_split_argument(self, split, language_pair):
        dataset1 = Multi30k(
            root=self.root_dir, split=split, language_pair=language_pair
        )
        (dataset2,) = Multi30k(
            root=self.root_dir, split=(split,), language_pair=language_pair
        )

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
