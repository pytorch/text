# -*- coding: utf-8 -*-
from unittest import TestCase
import json
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class TorchtextTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(format=('%(asctime)s - %(levelname)s - '
                                    '%(name)s - %(message)s'),
                            level=logging.INFO)
        # Directory where everything temporary and test-related is written
        self.test_dir = tempfile.mkdtemp()
        self.test_ppid_dataset_path = os.path.join(self.test_dir, "test_ppid_dataset")

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except:
            subprocess.call(["rm", "-rf", self.test_dir])

    def write_test_ppid_dataset(self, data_format="csv"):
        data_format = data_format.lower()
        if data_format == "csv":
            delim = ","
        elif data_format == "tsv":
            delim = "\t"
        dict_dataset = [
            {"id": "0", "question1": "When do you use シ instead of し?",
             "question2": "When do you use \"&\" instead of \"and\"?",
             "label": "0"},
            {"id": "1", "question1": "Where was Lincoln born?",
             "question2": "Which location was Abraham Lincoln born?",
             "label": "1"},
            {"id": "2", "question1": "What is 2+2",
             "question2": "2+2=?",
             "label": "1"},
        ]
        with open(self.test_ppid_dataset_path, "w") as test_ppid_dataset_file:
            for example in dict_dataset:
                if data_format == "json":
                    test_ppid_dataset_file.write(json.dumps(example) + "\n")
                elif data_format == "csv" or data_format == "tsv":
                    test_ppid_dataset_file.write("{}\n".format(
                        delim.join([example["id"], example["question1"],
                                    example["question2"], example["label"]])))
                else:
                    raise ValueError("Invalid format {}".format(data_format))
