import os
import zipfile
from collections import defaultdict
from unittest.mock import patch

from torchtext.datasets.wikitext103 import WikiText103
from torchtext.datasets.wikitext2 import WikiText2

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.parameterized_utils import nested_params
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir, base_dir_name):
    """
    root_dir: directory to the mocked dataset
    base_dir_name: WikiText103 or WikiText2
    """
    base_dir = os.path.join(root_dir, base_dir_name)
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)

    seed = 1
    mocked_data = defaultdict(list)
    file_names = ("wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens")
    for file_name in file_names:
        csv_file = os.path.join(temp_dataset_dir, file_name)
        mocked_lines = mocked_data[file_name.split(".")[1]]
        with open(csv_file, "w", encoding="utf-8") as f:
            for i in range(5):
                rand_string = get_random_unicode(seed)
                dataset_line = f"{rand_string}\n"
                f.write(dataset_line)

                # append line to correct dataset split
                mocked_lines.append(dataset_line)
                seed += 1

    if base_dir_name == WikiText103.__name__:
        compressed_file = "wikitext-103-v1"
        arcname_folder = "wikitext-103"
    else:
        compressed_file = "wikitext-2-v1"
        arcname_folder = "wikitext-2"

    compressed_dataset_path = os.path.join(base_dir, compressed_file + ".zip")
    # create zip file from dataset folder
    with zipfile.ZipFile(compressed_dataset_path, "w") as zip_file:
        for file_name in file_names:
            txt_file = os.path.join(temp_dataset_dir, file_name)
            zip_file.write(txt_file, arcname=os.path.join(arcname_folder, file_name))

    return mocked_data


class TestWikiTexts(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @ classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @ classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @ nested_params([WikiText103, WikiText2], ["train", "valid", "test"])
    def test_wikitexts(self, wikitext_dataset, split):
        expected_samples = _get_mock_dataset(self.root_dir, base_dir_name=wikitext_dataset.__name__)[split]

        dataset = wikitext_dataset(root=self.root_dir, split=split)
        samples = list(dataset)
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @ nested_params([WikiText103, WikiText2], ["train", "valid", "test"])
    def test_wikitexts_split_argument(self, wikitext_dataset, split):
        # call `_get_mock_dataset` to create mock dataset files
        _ = _get_mock_dataset(self.root_dir, wikitext_dataset.__name__)

        dataset1 = wikitext_dataset(root=self.root_dir, split=split)
        (dataset2,) = wikitext_dataset(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
