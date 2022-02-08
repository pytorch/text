import os
import random
import string
import tarfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.iwslt2016 import IWSLT2016, SUPPORTED_DATASETS
from torchtext.data.datasets_utils import _generate_iwslt_files_for_lang_and_split

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase

SUPPORTED_LANGPAIRS = [(k, e) for k, v in SUPPORTED_DATASETS["language_pair"].items() for e in v]


def _generate_uncleaned_train():
    """Generate tags files"""
    file_contents = []
    examples = []
    xml_tags = [
        '<url', '<keywords', '<talkid', '<description', '<reviewer',
        '<translator', '<title', '<speaker', '<doc', '</doc'
    ]
    for i in range(100):
        rand_string = " ".join(
            random.choice(string.ascii_letters) for i in range(10)
        )
        # With a 10% change, add one of the XML tags which is cleaned
        # to ensure cleaning happens appropriately
        if random.random() < 0.1:
            open_tag = random.choice(xml_tags) + ">"
            close_tag = "</" + open_tag[1:] + ">"
            file_contents.append(open_tag + rand_string + close_tag)
        else:
            examples.append(rand_string + "\n")
            file_contents.append(rand_string)
    return examples, "\n".join(file_contents)


def _generate_uncleaned_valid():
    file_contents = ["<root>"]
    examples = []

    for doc_id in range(5):
        file_contents.append(f'<doc docid="{doc_id}" genre="lectures">')
        for seg_id in range(100):
            rand_string = " ".join(
                random.choice(string.ascii_letters) for i in range(10)
            )
            examples.append(rand_string)
            file_contents.append(f"<seg>{rand_string} </seg>" + "\n")
        file_contents.append("</doc>")
    file_contents.append("</root>")
    return examples, " ".join(file_contents)


def _generate_uncleaned_test():
    return _generate_uncleaned_valid()


def _generate_uncleaned_contents(split):
    return {
        "train": _generate_uncleaned_train(),
        "valid": _generate_uncleaned_valid(),
        "test": _generate_uncleaned_test(),
    }[split]


def _get_mock_dataset(root_dir, split, src, tgt):
    """
    root_dir: directory to the mocked dataset
    """
    outer_temp_dataset_dir = os.path.join(root_dir, f"IWSLT2016/2016-01/texts/{src}/{tgt}/")
    inner_temp_dataset_dir = os.path.join(outer_temp_dataset_dir, f"{src}-{tgt}")

    os.makedirs(outer_temp_dataset_dir, exist_ok=True)
    os.makedirs(inner_temp_dataset_dir, exist_ok=True)

    mocked_data = defaultdict(lambda: defaultdict(list))
    valid_set = "tst2013"
    test_set = "tst2014"

    _, uncleaned_file_names = _generate_iwslt_files_for_lang_and_split(16, src, tgt, valid_set, test_set)
    src_file = uncleaned_file_names[src][split]
    tgt_file = uncleaned_file_names[tgt][split]
    for file_name in (src_file, tgt_file):
        out_file = os.path.join(inner_temp_dataset_dir, file_name)
        with open(out_file, "w") as f:
            # Get file extension (i.e., the language) without the . prefix (.en -> en)
            lang = os.path.splitext(file_name)[1][1:]
            mocked_data_for_split, file_contents = _generate_uncleaned_contents(split)
            mocked_data[split][lang] = mocked_data_for_split
            f.write(file_contents)

    inner_compressed_dataset_path = os.path.join(
        outer_temp_dataset_dir, f"{src}-{tgt}.tgz"
    )

    # create tar file from dataset folder
    with tarfile.open(inner_compressed_dataset_path, "w:gz") as tar:
        tar.add(inner_temp_dataset_dir, arcname=f"{src}-{tgt}")

    outer_temp_dataset_path = os.path.join(
        root_dir, "2016-01.tgz"
    )
    with tarfile.open(outer_temp_dataset_path, "w:gz") as tar:
        tar.add(outer_temp_dataset_dir, arcname="2016-01")

    return list(zip(mocked_data[split][src], mocked_data[split][tgt]))


class TestIWSLT2016(TempDirMixin, TorchtextTestCase):
    root_dir = None
    patcher = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True
        )
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand([
        (split, src, tgt)
        for split in ("train", "valid", "test")
        for src, tgt in SUPPORTED_LANGPAIRS
    ])
    def test_iwslt2016(self, split, src, tgt):
        expected_samples = _get_mock_dataset(self.root_dir, split, src, tgt)

        dataset = IWSLT2016(root=self.root_dir, split=split)

        samples = list(dataset)

        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "valid", "test"])
    def test_iwslt2016_split_argument(self, split):
        dataset1 = IWSLT2016(root=self.root_dir, split=split)
        (dataset2,) = IWSLT2016(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
