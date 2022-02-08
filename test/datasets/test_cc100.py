import os
import random
import string
import lzma
from parameterized import parameterized
from collections import defaultdict
from unittest.mock import patch

from torchtext.datasets import CC100

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase

from torchtext.datasets.cc100 import VALID_CODES


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "CC100")
    os.makedirs(base_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)

    for language_code in VALID_CODES:
        file_name = f"{language_code}.txt.xz"
        compressed_file = os.path.join(base_dir, file_name)
        with lzma.open(compressed_file, "wt") as f:
            for i in range(5):
                rand_string = " ".join(
                    random.choice(string.ascii_letters) for i in range(seed)
                )
                content = f"{rand_string}\n"
                f.write(content)
                mocked_data[language_code].append((language_code, rand_string))
                seed += 1

    return mocked_data


class TestCC100(TempDirMixin, TorchtextTestCase):
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

    @parameterized.expand(VALID_CODES)
    def test_cc100(self, language_code):
        dataset = CC100(root=self.root_dir, split="train", language_code=language_code)

        samples = list(dataset)
        expected_samples = self.samples[language_code]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)
