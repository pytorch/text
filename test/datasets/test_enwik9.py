import os
import zipfile
from unittest.mock import patch

from torchtext.datasets.enwik9 import EnWik9

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "EnWik9")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    file_name = "enwik9"
    txt_file = os.path.join(temp_dataset_dir, file_name)
    mocked_data = []
    with open(txt_file, "w", encoding="utf-8") as f:
        for i in range(5):
            rand_string = "<" + get_random_unicode(seed) + ">"
            dataset_line = f"'{rand_string}'"
            f.write(f"'{rand_string}'\n")

            # append line to correct dataset split
            mocked_data.append(dataset_line)
            seed += 1

    compressed_dataset_path = os.path.join(base_dir, "enwik9.zip")
    # create zip file from dataset folder
    with zipfile.ZipFile(compressed_dataset_path, "w") as zip_file:
        txt_file = os.path.join(temp_dataset_dir, file_name)
        zip_file.write(txt_file, arcname=file_name)

    return mocked_data


class TestEnWik9(TempDirMixin, TorchtextTestCase):
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

    def test_enwik9(self):
        dataset = EnWik9(root=self.root_dir)

        samples = list(dataset)
        expected_samples = self.samples
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)
