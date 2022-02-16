import json
import os
import uuid
from collections import defaultdict
from random import randint
from unittest.mock import patch

from torchtext.data.datasets_utils import _ParseSQuADQAData
from torchtext.datasets.squad1 import SQuAD1
from torchtext.datasets.squad2 import SQuAD2

from ..common.case_utils import TempDirMixin, zip_equal, get_random_unicode
from ..common.parameterized_utils import nested_params
from ..common.torchtext_test_case import TorchtextTestCase


def _get_mock_json_data():
    rand_string = get_random_unicode(10)
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


def _get_mock_dataset(root_dir, base_dir_name):
    """
    root_dir: directory to the mocked dataset
    """
    base_dir = os.path.join(root_dir, base_dir_name)
    os.makedirs(base_dir, exist_ok=True)

    if base_dir_name == SQuAD1.__name__:
        file_names = ("train-v1.1.json", "dev-v1.1.json")
    else:
        file_names = ("train-v2.0.json", "dev-v2.0.json")

    mocked_data = defaultdict(list)
    for file_name in file_names:
        txt_file = os.path.join(base_dir, file_name)
        with open(txt_file, "w", encoding="utf-8") as f:
            mock_json_data = _get_mock_json_data()
            f.write(json.dumps(mock_json_data))

            split = "train" if "train" in file_name else "dev"
            dataset_line = next(iter(_ParseSQuADQAData([("file_handle", mock_json_data)])))
            mocked_data[split].append(dataset_line)

    return mocked_data


class TestSQuADs(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @nested_params([SQuAD1, SQuAD2], ["train", "dev"])
    def test_squads(self, squad_dataset, split):
        expected_samples = _get_mock_dataset(self.root_dir, squad_dataset.__name__)[split]
        dataset = squad_dataset(root=self.root_dir, split=split)
        samples = list(dataset)

        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)

    @nested_params([SQuAD1, SQuAD2], ["train", "dev"])
    def test_squads_split_argument(self, squad_dataset, split):
        # call `_get_mock_dataset` to create mock dataset files
        _ = _get_mock_dataset(self.root_dir, squad_dataset.__name__)

        dataset1 = squad_dataset(root=self.root_dir, split=split)
        (dataset2,) = squad_dataset(root=self.root_dir, split=(split,))

        for d1, d2 in zip_equal(dataset1, dataset2):
            self.assertEqual(d1, d2)
