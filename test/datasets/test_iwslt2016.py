import itertools
import os
import random
import shutil
import string
import tarfile
import tempfile
from collections import defaultdict
from unittest.mock import patch

from parameterized import parameterized
from torchtext.data.datasets_utils import _generate_iwslt_files_for_lang_and_split
from torchtext.datasets.iwslt2016 import (
    DATASET_NAME,
    IWSLT2016,
    SUPPORTED_DATASETS,
    SET_NOT_EXISTS,
)

from ..common.case_utils import zip_equal
from ..common.torchtext_test_case import TorchtextTestCase

SUPPORTED_LANGPAIRS = [(k, e) for k, v in SUPPORTED_DATASETS["language_pair"].items() for e in v]
SUPPORTED_DEVTEST_SPLITS = SUPPORTED_DATASETS["valid_test"]
DEV_TEST_SPLITS = [(dev, test) for dev, test in itertools.product(SUPPORTED_DEVTEST_SPLITS, repeat=2) if dev != test]


def _generate_uncleaned_train():
    """Generate tags files"""
    file_contents = []
    examples = []
    xml_tags = [
        "<url",
        "<keywords",
        "<talkid",
        "<description",
        "<reviewer",
        "<translator",
        "<title",
        "<speaker",
        "<doc",
    ]
    for i in range(100):
        rand_string = " ".join(random.choice(string.ascii_letters) for i in range(10))
        # With a 10% change, add one of the XML tags which is cleaned
        # to ensure cleaning happens appropriately
        if random.random() < 0.1:
            open_tag = random.choice(xml_tags) + ">"
            # Open tag already contains the closing >
            close_tag = "</" + open_tag[1:]
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
            rand_string = " ".join(random.choice(string.ascii_letters) for i in range(10))
            examples.append(rand_string)
            file_contents.append(f"<seg>{rand_string} </seg>" + "\n")
        file_contents.append("</doc>")
    file_contents.append("</root>")
    return examples, " ".join(file_contents)


def _generate_uncleaned_test():
    return _generate_uncleaned_valid()


def _generate_uncleaned_contents(split):
    random.seed(1)
    return {
        "train": _generate_uncleaned_train(),
        "valid": _generate_uncleaned_valid(),
        "test": _generate_uncleaned_test(),
    }[split]


def _get_mock_dataset(root_dir, split, src, tgt, valid_set, test_set):
    """
    root_dir: directory to the mocked dataset
    """

    base_dir = os.path.join(root_dir, DATASET_NAME)
    temp_dataset_dir = os.path.join(base_dir, "temp_dataset_dir")
    outer_temp_dataset_dir = os.path.join(temp_dataset_dir, f"texts/{src}/{tgt}/")
    inner_temp_dataset_dir = os.path.join(outer_temp_dataset_dir, f"{src}-{tgt}")

    os.makedirs(outer_temp_dataset_dir, exist_ok=True)
    os.makedirs(inner_temp_dataset_dir, exist_ok=True)

    mocked_data = defaultdict(lambda: defaultdict(list))

    cleaned_file_names, uncleaned_file_names = _generate_iwslt_files_for_lang_and_split(
        16, src, tgt, valid_set, test_set
    )
    uncleaned_src_file = uncleaned_file_names[src][split]
    uncleaned_tgt_file = uncleaned_file_names[tgt][split]

    cleaned_src_file = cleaned_file_names[src][split]
    cleaned_tgt_file = cleaned_file_names[tgt][split]

    for (unclean_file_name, clean_file_name) in [
        (uncleaned_src_file, cleaned_src_file),
        (uncleaned_tgt_file, cleaned_tgt_file),
    ]:
        # Get file extension (i.e., the language) without the . prefix (.en -> en)
        lang = os.path.splitext(unclean_file_name)[1][1:]

        out_file = os.path.join(inner_temp_dataset_dir, unclean_file_name)
        with open(out_file, "w") as f:
            mocked_data_for_split, file_contents = _generate_uncleaned_contents(split)
            mocked_data[split][lang] = mocked_data_for_split
            f.write(file_contents)

    inner_compressed_dataset_path = os.path.join(outer_temp_dataset_dir, f"{src}-{tgt}.tgz")

    # create tar file from dataset folder
    with tarfile.open(inner_compressed_dataset_path, "w:gz") as tar:
        tar.add(inner_temp_dataset_dir, arcname=f"{src}-{tgt}")

    # this is necessary so that the outer tarball only includes the inner tarball
    shutil.rmtree(inner_temp_dataset_dir)

    outer_temp_dataset_path = os.path.join(base_dir, "2016-01.tgz")

    with tarfile.open(outer_temp_dataset_path, "w:gz") as tar:
        tar.add(temp_dataset_dir, arcname="2016-01")

    return list(zip(mocked_data[split][src], mocked_data[split][tgt]))


class TestIWSLT2016(TorchtextTestCase):
    root_dir = None
    patcher = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(
        [
            (split, src, tgt, dev_set, test_set)
            for split in ("train", "valid", "test")
            for dev_set, test_set in DEV_TEST_SPLITS
            for src, tgt in SUPPORTED_LANGPAIRS
            if (dev_set not in SET_NOT_EXISTS[(src, tgt)] and test_set not in SET_NOT_EXISTS[(src, tgt)])
        ]
    )
    def test_iwslt2016(self, split, src, tgt, dev_set, test_set):

        with tempfile.TemporaryDirectory() as root_dir:
            expected_samples = _get_mock_dataset(os.path.join(root_dir, "datasets"), split, src, tgt, dev_set, test_set)

            dataset = IWSLT2016(
                root=root_dir,
                split=split,
                language_pair=(src, tgt),
                valid_set=dev_set,
                test_set=test_set,
            )

            samples = list(dataset)

            for sample, expected_sample in zip_equal(samples, expected_samples):
                self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "valid", "test"])
    def test_iwslt2016_split_argument(self, split):
        with tempfile.TemporaryDirectory() as root_dir:
            language_pair = ("de", "en")
            valid_set = "tst2013"
            test_set = "tst2014"
            _ = _get_mock_dataset(
                os.path.join(root_dir, "datasets"), split, language_pair[0], language_pair[1], valid_set, test_set
            )
            dataset1 = IWSLT2016(
                root=root_dir,
                split=split,
                language_pair=language_pair,
                valid_set=valid_set,
                test_set=test_set,
            )
            (dataset2,) = IWSLT2016(
                root=root_dir,
                split=(split,),
                language_pair=language_pair,
                valid_set=valid_set,
                test_set=test_set,
            )

            for d1, d2 in zip_equal(dataset1, dataset2):
                self.assertEqual(d1, d2)
