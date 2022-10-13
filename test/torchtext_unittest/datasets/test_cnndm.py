import hashlib
import os
import tarfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets import CNNDM

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """

    base_dir = os.path.join(root_dir, "CNNDM")
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    os.makedirs(temp_dataset_dir, exist_ok=True)
    seed = 1
    mocked_data = defaultdict(list)

    for source in ["cnn", "dailymail"]:
        source_dir = os.path.join(temp_dataset_dir, source, "stories")
        os.makedirs(source_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            stories = []
            for i in range(5):
                url = "_".join([source, split, str(i)])
                h = hashlib.sha1()
                h.update(url.encode())
                filename = h.hexdigest() + ".story"
                txt_file = os.path.join(source_dir, filename)
                with open(txt_file, "w", encoding=("utf-8")) as f:
                    article = get_random_unicode(seed) + "."
                    abstract = get_random_unicode(seed + 1) + "."
                    dataset_line = (article, abstract)
                    f.writelines([article, "\n@highlight\n", abstract])
                    stories.append((txt_file, dataset_line))
                seed += 2

            # append stories to correct dataset split, must be in lexicographic order of filenames per dataset
            stories.sort(key=lambda x: x[0])
            mocked_data[split] += [t[1] for t in stories]

        compressed_dataset_path = os.path.join(base_dir, f"{source}_stories.tgz")
        # create zip file from dataset folder
        with tarfile.open(compressed_dataset_path, "w:gz") as tar:
            tar.add(os.path.join(temp_dataset_dir, source), arcname=source)

    return mocked_data


class TestCNNDM(TempDirMixin, TorchtextTestCase):
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

    def _mock_split_list(source, split):
        story_fnames = []
        for i in range(5):
            url = "_".join([source, split, str(i)])
            h = hashlib.sha1()
            h.update(url.encode())
            filename = h.hexdigest() + ".story"
            story_fnames.append(filename)

        return story_fnames

    @parameterized.expand(["train", "val", "test"])
    @patch("torchtext.datasets.cnndm._get_split_list", _mock_split_list)
    def test_cnndm(self, split):
        dataset = CNNDM(root=self.root_dir, split=split)
        samples = list(dataset)
        expected_samples = self.samples[split]
        self.assertEqual(expected_samples, samples)

    @parameterized.expand(["train", "val", "test"])
    @patch("torchtext.datasets.cnndm._get_split_list", _mock_split_list)
    def test_cnndm_split_argument(self, split):
        dataset1 = CNNDM(root=self.root_dir, split=split)
        (dataset2,) = CNNDM(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
