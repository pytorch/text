# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import torchtext.data as data

import pytest

from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    def test_tabular_simple_data(self):
        for data_format in ["csv", "tsv", "json"]:
            self.write_test_ppid_dataset(data_format=data_format)

            if data_format == "json":
                question_field = data.Field(sequential=True)
                label_field = data.Field(sequential=False)
                fields = {"question1": ("q1", question_field),
                          "question2": ("q2", question_field),
                          "label": ("label", label_field)}
            else:
                question_field = data.Field(sequential=True)
                label_field = data.Field(sequential=False)
                fields = [("id", None), ("q1", question_field),
                          ("q2", question_field), ("label", label_field)]

            dataset = data.TabularDataset(
                path=self.test_ppid_dataset_path, format=data_format, fields=fields)

            assert len(dataset) == 3

            expected_examples = [
                (["When", "do", "you", "use", "シ", "instead", "of", "し?"],
                 ["When", "do", "you", "use", "\"&\"",
                  "instead", "of", "\"and\"?"], "0"),
                (["Where", "was", "Lincoln", "born?"],
                 ["Which", "location", "was", "Abraham", "Lincoln", "born?"], "1"),
                (["What", "is", "2+2"], ["2+2=?"], "1")]

            # Ensure examples have correct contents / test __getitem__
            for i in range(len(dataset)):
                self.assertEqual(dataset[i].q1, expected_examples[i][0])
                self.assertEqual(dataset[i].q2, expected_examples[i][1])
                self.assertEqual(dataset[i].label, expected_examples[i][2])

            # Test __getattr__
            for i, (q1, q2, label) in enumerate(zip(dataset.q1, dataset.q2,
                                                    dataset.label)):
                self.assertEqual(q1, expected_examples[i][0])
                self.assertEqual(q2, expected_examples[i][1])
                self.assertEqual(label, expected_examples[i][2])

            # Test __iter__
            for i, example in enumerate(dataset):
                self.assertEqual(example.q1, expected_examples[i][0])
                self.assertEqual(example.q2, expected_examples[i][1])
                self.assertEqual(example.label, expected_examples[i][2])

    def test_json_dataset_one_key_multiple_fields(self):
        self.write_test_ppid_dataset(data_format="json")

        question_field = data.Field(sequential=True)
        spacy_tok_question_field = data.Field(sequential=True, tokenize="spacy")
        label_field = data.Field(sequential=False)
        fields = {"question1": [("q1", question_field),
                                ("q1_spacy", spacy_tok_question_field)],
                  "question2": [("q2", question_field),
                                ("q2_spacy", spacy_tok_question_field)],
                  "label": ("label", label_field)}
        dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="json", fields=fields)
        expected_examples = [
            (["When", "do", "you", "use", "シ", "instead", "of", "し?"],
             ["When", "do", "you", "use", "シ", "instead", "of", "し", "?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"],
             ["When", "do", "you", "use", "\"", "&", "\"",
              "instead", "of", "\"", "and", "\"", "?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Where", "was", "Lincoln", "born", "?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born", "?"],
             "1"),
            (["What", "is", "2+2"], ["What", "is", "2", "+", "2"],
             ["2+2=?"], ["2", "+", "2=", "?"], "1")]
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q1_spacy, expected_examples[i][1])
            self.assertEqual(example.q2, expected_examples[i][2])
            self.assertEqual(example.q2_spacy, expected_examples[i][3])
            self.assertEqual(example.label, expected_examples[i][4])

    def test_errors(self):
        # Ensure that trying to retrieve a key not in JSON data errors
        self.write_test_ppid_dataset(data_format="json")

        question_field = data.Field(sequential=True)
        label_field = data.Field(sequential=False)
        fields = {"qeustion1": ("q1", question_field),
                  "question2": ("q2", question_field),
                  "label": ("label", label_field)}

        with self.assertRaises(ValueError):
            data.TabularDataset(
                path=self.test_ppid_dataset_path, format="json", fields=fields)

    def test_input_with_newlines_in_text(self):
        # Smoke test for ensuring that TabularDataset works with files with newlines
        example_with_newlines = [("\"hello \n world\"", "1"),
                                 ("\"there is a \n newline\"", "0"),
                                 ("\"there is no newline\"", "1")]
        fields = [("text", data.Field(lower=True)),
                  ("label", data.Field(sequential=False))]

        for delim in [",", "\t"]:
            with open(self.test_newline_dataset_path, "wt") as f:
                for line in example_with_newlines:
                    f.write("{}\n".format(delim.join(line)))

            format_ = "csv" if delim == "," else "tsv"
            dataset = data.TabularDataset(
                path=self.test_newline_dataset_path, format=format_, fields=fields)
            # if the newline is not parsed correctly, this should raise an error
            for example in dataset:
                self.assert_(hasattr(example, "text"))
                self.assert_(hasattr(example, "label"))

    def test_csv_file_with_header(self):
        example_with_header = [("text", "label"),
                               ("HELLO WORLD", "0"),
                               ("goodbye world", "1")]

        TEXT = data.Field(lower=True, tokenize=lambda x: x.split())
        fields = {
            "label": ("label", data.Field(use_vocab=False,
                                          sequential=False)),
            "text": ("text", TEXT)
        }

        for format_, delim in zip(["csv", "tsv"], [",", "\t"]):
            with open(self.test_has_header_dataset_path, "wt") as f:
                for line in example_with_header:
                    f.write("{}\n".format(delim.join(line)))

            # check that an error is raised here if a non-existent field is specified
            with self.assertRaises(ValueError):
                data.TabularDataset(
                    path=self.test_has_header_dataset_path, format=format_,
                    fields={"non_existent": ("label", data.Field())})

            dataset = data.TabularDataset(
                path=self.test_has_header_dataset_path, format=format_,
                skip_header=False, fields=fields)

            TEXT.build_vocab(dataset)

            for i, example in enumerate(dataset):
                self.assertEqual(example.text,
                                 example_with_header[i + 1][0].lower().split())
                self.assertEqual(example.label, example_with_header[i + 1][1])

            # check that the vocabulary is built correctly (#225)
            expected_freqs = {"hello": 1, "world": 2, "goodbye": 1, "text": 0}
            for k, v in expected_freqs.items():
                self.assertEqual(TEXT.vocab.freqs[k], v)

            data_iter = data.Iterator(dataset, device=-1, batch_size=1,
                                      sort_within_batch=False, repeat=False)
            next(data_iter.__iter__())

    def test_csv_file_no_header_one_col_multiple_fields(self):
        self.write_test_ppid_dataset(data_format="csv")

        question_field = data.Field(sequential=True)
        spacy_tok_question_field = data.Field(sequential=True, tokenize="spacy")
        label_field = data.Field(sequential=False)
        # Field name/value as nested tuples
        fields = [("ids", None),
                  (("q1", "q1_spacy"), (question_field, spacy_tok_question_field)),
                  (("q2", "q2_spacy"), (question_field, spacy_tok_question_field)),
                  ("label", label_field)]
        dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="csv", fields=fields)
        expected_examples = [
            (["When", "do", "you", "use", "シ", "instead", "of", "し?"],
             ["When", "do", "you", "use", "シ", "instead", "of", "し", "?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"],
             ["When", "do", "you", "use", "\"", "&", "\"",
              "instead", "of", "\"", "and", "\"", "?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Where", "was", "Lincoln", "born", "?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born", "?"],
             "1"),
            (["What", "is", "2+2"], ["What", "is", "2", "+", "2"],
             ["2+2=?"], ["2", "+", "2=", "?"], "1")]
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q1_spacy, expected_examples[i][1])
            self.assertEqual(example.q2, expected_examples[i][2])
            self.assertEqual(example.q2_spacy, expected_examples[i][3])
            self.assertEqual(example.label, expected_examples[i][4])

        # 6 Fields including None for ids
        assert len(dataset.fields) == 6

    def test_dataset_split_arguments(self):
        num_examples, num_labels = 30, 3
        self.write_test_splitting_dataset(num_examples=num_examples,
                                          num_labels=num_labels)
        text_field = data.Field()
        label_field = data.LabelField()
        fields = [('text', text_field), ('label', label_field)]

        dataset = data.TabularDataset(
            path=self.test_dataset_splitting_path, format="csv", fields=fields)

        # Test default split ratio (0.7)
        expected_train_size = 21
        expected_test_size = 9

        train, test = dataset.split()
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        # Test array arguments with same ratio
        split_ratio = [0.7, 0.3]
        train, test = dataset.split(split_ratio=split_ratio)
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        # Add validation set
        split_ratio = [0.6, 0.3, 0.1]
        expected_train_size = 18
        expected_valid_size = 3
        expected_test_size = 9

        train, valid, test = dataset.split(split_ratio=split_ratio)
        assert len(train) == expected_train_size
        assert len(valid) == expected_valid_size
        assert len(test) == expected_test_size

        # Test ratio normalization
        split_ratio = [6, 3, 1]
        train, valid, test = dataset.split(split_ratio=split_ratio)
        assert len(train) == expected_train_size
        assert len(valid) == expected_valid_size
        assert len(test) == expected_test_size

        # Test only two splits returned for too small valid split size
        split_ratio = [0.66, 0.33, 0.01]
        expected_length = 2
        splits = dataset.split(split_ratio=split_ratio)
        assert len(splits) == expected_length

        # Test invalid arguments
        split_ratio = 1.1
        with pytest.raises(AssertionError):
            dataset.split(split_ratio=split_ratio)

        split_ratio = -1.
        with pytest.raises(AssertionError):
            dataset.split(split_ratio=split_ratio)

        split_ratio = [0.7]
        with pytest.raises(AssertionError):
            dataset.split(split_ratio=split_ratio)

        split_ratio = [1, 2, 3, 4]
        with pytest.raises(AssertionError):
            dataset.split(split_ratio=split_ratio)

        split_ratio = "string"
        with pytest.raises(ValueError):
            dataset.split(split_ratio=split_ratio)

    def test_stratified_dataset_split(self):
        num_examples, num_labels = 30, 3
        self.write_test_splitting_dataset(num_examples=num_examples,
                                          num_labels=num_labels)
        text_field = data.Field()
        label_field = data.LabelField()
        fields = [('text', text_field), ('label', label_field)]

        dataset = data.TabularDataset(
            path=self.test_dataset_splitting_path, format="csv", fields=fields)

        # Default split ratio
        expected_train_size = 21
        expected_test_size = 9

        train, test = dataset.split(stratified=True)
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        # Test array arguments with same ratio
        split_ratio = [0.7, 0.3]
        train, test = dataset.split(split_ratio=split_ratio, stratified=True)
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        # Test strata_field argument
        train, test = dataset.split(split_ratio=split_ratio, stratified=True,
                                    strata_field='label')
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        # Test invalid field name
        strata_field = 'dummy'
        with pytest.raises(ValueError):
            dataset.split(split_ratio=split_ratio, stratified=True,
                          strata_field=strata_field)

        # Test uneven stratify sizes
        num_examples, num_labels = 28, 3
        self.write_test_splitting_dataset(num_examples=num_examples,
                                          num_labels=num_labels)
        # 10 examples for class 1 and 9 examples for classes 2,3
        dataset = data.TabularDataset(
            path=self.test_dataset_splitting_path, format="csv", fields=fields)

        expected_train_size = 7 + 6 + 6
        expected_test_size = 3 + 3 + 3
        train, test = dataset.split(split_ratio=split_ratio, stratified=True)
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        split_ratio = [0.7, 0.3]
        train, test = dataset.split(split_ratio=split_ratio, stratified=True)
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size

        # Add validation set
        split_ratio = [0.6, 0.3, 0.1]
        expected_train_size = 6 + 5 + 5
        expected_valid_size = 1 + 1 + 1
        expected_test_size = 3 + 3 + 3
        train, valid, test = dataset.split(split_ratio=split_ratio, stratified=True)
        assert len(train) == expected_train_size
        assert len(valid) == expected_valid_size
        assert len(test) == expected_test_size
