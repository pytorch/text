import json
import os
import random
import string
import uuid
from collections import defaultdict
from random import randint
from unittest.mock import patch

from parameterized import parameterized
from torchtext.data.datasets_utils import _ParseSQuADQAData
from torchtext.datasets.squad2 import SQuAD2

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_json_data():
    rand_string = " ".join(random.choice(string.ascii_letters) for i in range(10))
    mock_json_data = {
        "data": [
            {
                "title": rand_string,
                "paragraphs": [
                    {
                        "context": rand_string,
                        "qas": [
                            {
                                "answers": [
                                    {
                                        "answer_start": randint(1, 1000),
                                        "text": rand_string,
                                    }
                                ],
                                "question": rand_string,
                                "id": uuid.uuid1().hex,
                            },
                        ],
                    }
                ],
            }
        ]
    }
    return mock_json_data


def _get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, "SQuAD2")
    os.makedirs(base_dir, exist_ok=True)

    mocked_data = defaultdict(list)
    for file_name in ("train-v2.0.json", "dev-v2.0.json"):
        txt_file = os.path.join(base_dir, file_name)
        with open(txt_file, "w") as f:
            mock_json_data = _get_mock_json_data()
            f.write(json.dumps(mock_json_data))

            split = "train" if "train" in file_name else "dev"
            dataset_line = next(
                iter(_ParseSQuADQAData([("file_handle", mock_json_data)]))
            )
            mocked_data[split].append(dataset_line)

    return mocked_data


class TestSQuAD1(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

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

    @parameterized.expand(["train", "dev"])
    def test_squad2(self, split):
        dataset = SQuAD2(root=self.root_dir, split=split)

        samples = list(dataset)
        expected_samples = self.samples[split]
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @parameterized.expand(["train", "dev"])
    def test_squad2_split_argument(self, split):
        dataset1 = SQuAD2(root=self.root_dir, split=split)
        (dataset2,) = SQuAD2(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
