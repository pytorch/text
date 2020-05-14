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
        self.project_root = os.path.abspath(os.path.realpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)))
        self.test_dir = tempfile.mkdtemp()
        self.test_ppid_dataset_path = os.path.join(self.test_dir, "test_ppid_dataset")
        self.test_numerical_features_dataset_path = os.path.join(
            self.test_dir, "test_numerical_features_dataset")
        self.test_newline_dataset_path = os.path.join(self.test_dir,
                                                      "test_newline_dataset")
        self.test_has_header_dataset_path = os.path.join(self.test_dir,
                                                         "test_has_header_dataset")
        self.test_missing_field_dataset_path = os.path.join(self.test_dir,
                                                            "test_msg_field_dst")
        self.test_dataset_splitting_path = os.path.join(self.test_dir,
                                                        "test_dataset_split")
        self.test_nested_key_json_dataset_path = os.path.join(self.test_dir,
                                                              "test_nested_key_json")

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
        with open(self.test_ppid_dataset_path, "w", encoding="utf-8") as test_ppid_dataset_file:
            for example in dict_dataset:
                if data_format == "json":
                    test_ppid_dataset_file.write(json.dumps(example) + "\n")
                elif data_format == "csv" or data_format == "tsv":
                    test_ppid_dataset_file.write("{}\n".format(
                        delim.join([example["id"], example["question1"],
                                    example["question2"], example["label"]])))
                else:
                    raise ValueError("Invalid format {}".format(data_format))

    def write_test_nested_key_json_dataset(self):
        """
        Used only to test nested key parsing of Example.fromJSON()
        """
        dict_dataset = [
            {"foods":
                {"fruits": ["Apple", "Banana"],
                 "vegetables": [
                    {"name": "Broccoli"},
                    {"name": "Cabbage"}]}},
            {"foods":
                {"fruits": ["Cherry", "Grape", "Lemon"],
                 "vegetables": [
                    {"name": "Cucumber"},
                    {"name": "Lettuce"}]}},
            {"foods":
                {"fruits": ["Orange", "Pear", "Strawberry"],
                 "vegetables": [
                    {"name": "Marrow"},
                    {"name": "Spinach"}]}},
        ]
        with open(self.test_nested_key_json_dataset_path,
                  "w") as test_nested_key_json_dataset_file:
            for example in dict_dataset:
                test_nested_key_json_dataset_file.write(json.dumps(example) + "\n")

    def write_test_numerical_features_dataset(self):
        with open(self.test_numerical_features_dataset_path,
                  "w") as test_numerical_features_dataset_file:
            test_numerical_features_dataset_file.write("0.1\t1\tteststring1\n")
            test_numerical_features_dataset_file.write("0.5\t12\tteststring2\n")
            test_numerical_features_dataset_file.write("0.2\t0\tteststring3\n")
            test_numerical_features_dataset_file.write("0.4\t12\tteststring4\n")
            test_numerical_features_dataset_file.write("0.9\t9\tteststring5\n")

    def make_mock_dataset(self, num_examples=30, num_labels=3):
        num_repetitions = int(round(num_examples / num_labels)) + 1

        texts = [str(i) for i in range(num_examples)]
        labels = list(range(num_labels)) * num_repetitions
        labels = [str(l) for l in labels[:num_examples]]

        dict_dataset = [
            {'text': t, 'label': l} for t, l in zip(texts, labels)
        ]
        return dict_dataset

    def write_test_splitting_dataset(self, num_examples=30, num_labels=3):
        dict_dataset = self.make_mock_dataset(num_examples, num_labels)
        delim = ","

        with open(self.test_dataset_splitting_path,
                  "w") as test_splitting_dataset_file:
            for example in dict_dataset:
                test_splitting_dataset_file.write("{}\n".format(
                    delim.join([example['text'], example['label']])))


def verify_numericalized_example(field, test_example_data,
                                 test_example_numericalized,
                                 test_example_lengths=None,
                                 batch_first=False, train=True):
    """
    Function to verify that numericalized example is correct
    with respect to the Field's Vocab.
    """
    if isinstance(test_example_numericalized, tuple):
        test_example_numericalized, lengths = test_example_numericalized
        assert test_example_lengths == lengths.tolist()
    if batch_first:
        test_example_numericalized.t_()
    # Transpose numericalized example so we can compare over batches
    for example_idx, numericalized_single_example in enumerate(
            test_example_numericalized.t()):
        assert len(test_example_data[example_idx]) == len(numericalized_single_example)
        assert numericalized_single_example.volatile is not train
        for token_idx, numericalized_token in enumerate(
                numericalized_single_example):
            # Convert from Variable to int
            numericalized_token = numericalized_token.item()  # Pytorch v4 compatibility
            test_example_token = test_example_data[example_idx][token_idx]
            # Check if the numericalized example is correct, taking into
            # account unknown tokens.
            if field.vocab.stoi[test_example_token] != 0:
                # token is in-vocabulary
                assert (field.vocab.itos[numericalized_token]
                        == test_example_token)
            else:
                # token is OOV and <unk> always has an index of 0
                assert numericalized_token == 0
