import os
import random
import string
import tarfile
import itertools
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.datasets.iwslt2016 import IWSLT2016, SUPPORTED_DATASETS, SET_NOT_EXISTS
from torchtext.data.datasets_utils import _generate_iwslt_files_for_lang_and_split

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase

SUPPORTED_LANGPAIRS = [(k, e) for k, v in SUPPORTED_DATASETS["language_pair"].items() for e in v]
SUPPORTED_DEVTEST_SPLITS = SUPPORTED_DATASETS["valid_test"]
DEV_TEST_SPLITS = [(dev, test) for dev, test in itertools.product(SUPPORTED_DEVTEST_SPLITS, repeat=2) if dev != test]


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


def _get_mock_dataset(root_dir, split, src, tgt, valid_set, test_set):
    """
    root_dir: directory to the mocked dataset
    """
    outer_temp_dataset_dir = os.path.join(root_dir, f"IWSLT2016/2016-01/texts/{src}/{tgt}/")
    inner_temp_dataset_dir = os.path.join(outer_temp_dataset_dir, f"{src}-{tgt}")

    os.makedirs(outer_temp_dataset_dir, exist_ok=True)
    os.makedirs(inner_temp_dataset_dir, exist_ok=True)

    mocked_data = defaultdict(lambda: defaultdict(list))

    cleaned_file_names, uncleaned_file_names = _generate_iwslt_files_for_lang_and_split(16, src, tgt, valid_set, test_set)
    uncleaned_src_file = uncleaned_file_names[src][split]
    uncleaned_tgt_file = uncleaned_file_names[tgt][split]

    cleaned_src_file = cleaned_file_names[src][split]
    cleaned_tgt_file = cleaned_file_names[tgt][split]

    for (unclean_file_name, clean_file_name) in [
        (uncleaned_src_file, cleaned_src_file),
        (uncleaned_tgt_file, cleaned_tgt_file)
    ]:
        # Get file extension (i.e., the language) without the . prefix (.en -> en)
        lang = os.path.splitext(unclean_file_name)[1][1:]
        expected_clean_filename = os.path.join(inner_temp_dataset_dir, clean_file_name)

        # If we've already written a clean file, read it, so we don't generate
        # new random strings. Otherwise generate new files and clean when read.
        if os.path.exists(expected_clean_filename):
            with open(expected_clean_filename, encoding="utf-8") as f:
                mocked_data[(split, valid_set, test_set)][lang] = f.readlines()
        else:
            out_file = os.path.join(inner_temp_dataset_dir, unclean_file_name)
            with open(out_file, "w") as f:
                mocked_data_for_split, file_contents = _generate_uncleaned_contents(split)
                mocked_data[(split, valid_set, test_set)][lang] = mocked_data_for_split
                f.write(file_contents)

    inner_compressed_dataset_path = os.path.join(
        outer_temp_dataset_dir, f"{src}-{tgt}.tgz"
    )

    # create tar file from dataset folder
    with tarfile.open(inner_compressed_dataset_path, "w:gz") as tar:
        tar.add(inner_temp_dataset_dir, arcname=f"{src}-{tgt}")

    outer_temp_dataset_path = os.path.join(
        root_dir, "IWSLT2016", "2016-01.tgz"
    )
    with tarfile.open(outer_temp_dataset_path, "w:gz") as tar:
        tar.add(outer_temp_dataset_dir, arcname="2016-01")

    return list(zip(mocked_data[(split, valid_set, test_set)][src], mocked_data[(split, valid_set, test_set)][tgt]))


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
        (split, src, tgt, dev_set, test_set)
        for split in ("train", "valid", "test")
        for dev_set, test_set in DEV_TEST_SPLITS
        for src, tgt in SUPPORTED_LANGPAIRS
        if (dev_set not in SET_NOT_EXISTS[(src, tgt)] and test_set not in SET_NOT_EXISTS[(src, tgt)])
    ])
    def test_iwslt2016(self, split, src, tgt, dev_set, test_set):

        expected_samples = _get_mock_dataset(self.root_dir, split, src, tgt, dev_set, test_set)

        dataset = IWSLT2016(
            root=self.root_dir, split=split, language_pair=(src, tgt), valid_set=dev_set, test_set=test_set
        )

        samples = list(dataset)

        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "valid", "test"])
    def test_iwslt2016_split_argument(self, split):
        dataset1 = IWSLT2016(root=self.root_dir, split=split)
        (dataset2,) = IWSLT2016(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
