import os
import random
import string
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.data.datasets_utils import _generate_iwslt_files_for_lang_and_split
from torchtext.datasets.iwslt2016 import IWSLT2016

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir, split, src, tgt):
    """
    root_dir: directory to the mocked dataset
    """
    temp_dataset_dir = os.path.join(root_dir, f"IWSLT2016/2016-01/texts/{src}/{tgt}/{src}-{tgt}/")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(lambda: defaultdict(list))
    valid_set = "tst2013"
    test_set = "tst2014"

    files_for_split, _ = _generate_iwslt_files_for_lang_and_split(16, src, tgt, valid_set, test_set)
    src_file = files_for_split[src][split]
    tgt_file = files_for_split[tgt][split]
    for file_name in (src_file, tgt_file):
        txt_file = os.path.join(temp_dataset_dir, file_name)
        with open(txt_file, "w") as f:
            # Get file extension (i.e., the language) without the . prefix (.en -> en)
            lang = os.path.splitext(file_name)[1][1:]
            for i in range(5):
                rand_string = " ".join(random.choice(string.ascii_letters) for i in range(seed))
                dataset_line = f"{rand_string} {rand_string}\n"
                # append line to correct dataset split
                mocked_data[split][lang].append(dataset_line)
                f.write(f"{rand_string} {rand_string}\n")
                seed += 1

    return list(zip(mocked_data[split][src], mocked_data[split][tgt]))


class TestIWSLT2016(TempDirMixin, TorchtextTestCase):
    root_dir = None
    patcher = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder.OnDiskCacheHolderIterDataPipe._cache_check_fn", return_value=True
        )
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand([("train", "de", "en"), ("valid", "de", "en")])
    def test_iwslt2016(self, split, src, tgt):
        expected_samples = _get_mock_dataset(self.root_dir, split, src, tgt)

        dataset = IWSLT2016(root=self.root_dir, split=split)

        samples = list(dataset)

        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "valid"])
    def test_iwslt2016_split_argument(self, split):
        dataset1 = IWSLT2016(root=self.root_dir, split=split)
        (dataset2,) = IWSLT2016(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
