import os
from unittest.mock import patch

from torchtext.datasets.qqp import QQP

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "QQP")
    os.makedirs(base_dir, exist_ok=True)

    seed = 1
    file_name = "quora_duplicate_questions.tsv"
    txt_file = os.path.join(base_dir, file_name)
    mocked_data = []
    print(txt_file)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("id\tqid1\tqid2\tquestion1\tquestion2\tis_duplicate\n")
        for i in range(5):
            label = seed % 2
            rand_string_1 = get_random_unicode(seed)
            rand_string_2 = get_random_unicode(seed + 1)
            dataset_line = (label, rand_string_1, rand_string_2)
            # append line to correct dataset split
            mocked_data.append(dataset_line)
            f.write(f"{i}\t{i}\t{i}\t{rand_string_1}\t{rand_string_2}\t{label}\n")
            seed += 1

    return mocked_data


class TestQQP(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        print(cls.root_dir)
        cls.samples = _get_mock_dataset(cls.root_dir)
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    def test_qqp(self):
        dataset = QQP(root=self.root_dir)

        samples = list(dataset)
        expected_samples = self.samples
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)
