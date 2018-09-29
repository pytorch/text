# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import torchtext.data as data
import tempfile
import six

import pytest

from ..common.torchtext_test_case import TorchtextTestCase
import os


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

            data_iter = data.Iterator(dataset, batch_size=1,
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

    def test_csv_dataset_quotechar(self):
        # Based on issue #349
        example_data = [("text", "label"),
                        ('" hello world', "0"),
                        ('goodbye " world', "1"),
                        ('this is a pen " ', "0")]

        with tempfile.NamedTemporaryFile(dir=self.test_dir) as f:
            for example in example_data:
                f.write(six.b("{}\n".format(",".join(example))))

            TEXT = data.Field(lower=True, tokenize=lambda x: x.split())
            fields = {
                "label": ("label", data.Field(use_vocab=False,
                                              sequential=False)),
                "text": ("text", TEXT)
            }

            f.seek(0)

            dataset = data.TabularDataset(
                path=f.name, format="csv",
                skip_header=False, fields=fields,
                csv_reader_params={"quotechar": None})

            TEXT.build_vocab(dataset)

            self.assertEqual(len(dataset), len(example_data) - 1)

            for i, example in enumerate(dataset):
                self.assertEqual(example.text,
                                 example_data[i + 1][0].lower().split())
                self.assertEqual(example.label, example_data[i + 1][1])

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

    def test_filter(self):
        # Create test examples
        sentence11 = [["who", "is", "there"]]
        sentence12 = [["bernardo", "is", "there"]]
        label1 = [1]
        sentence21 = [["nay", "answer", "me"]]
        sentence22 = [["stand", "unfold", "yourself"]]
        label2 = [0]
        sentence31 = [["is", "Horatio", "there"]]
        sentence32 = [["a", "piece", "of", "him"]]
        label3 = [0]

        example1_values = sentence11 + sentence12 + label1
        example2_values = sentence21 + sentence22 + label2
        example3_values = sentence31 + sentence32 + label3

        # Test filter remove words from single field only
        dataset, text_field = filter_init(
            example1_values,
            example2_values,
            example3_values
        )

        text_field.vocab.stoi.pop("there")
        text_field.vocab.stoi.pop("bernardo")

        dataset.filter_examples(["text1"])

        assert dataset[0].text1 == ["who", "is"]
        assert dataset[0].text2 == ["bernardo", "is", "there"]
        assert dataset[0].label == 1

        assert dataset[1].text1 == ["nay", "answer", "me"]
        assert dataset[1].text2 == ["stand", "unfold", "yourself"]
        assert dataset[1].label == 0

        assert dataset[2].text1 == ["is", "Horatio"]
        assert dataset[2].text2 == ["a", "piece", "of", "him"]
        assert dataset[2].label == 0

        # Test filter remove words from multiple fields
        dataset, text_field = filter_init(
            example1_values,
            example2_values,
            example3_values
        )

        text_field.vocab.stoi.pop("there")
        text_field.vocab.stoi.pop("bernardo")

        dataset.filter_examples(["text1", "text2"])

        assert dataset[0].text1 == ["who", "is"]
        assert dataset[0].text2 == ["is"]
        assert dataset[0].label == 1

        assert dataset[1].text1 == ["nay", "answer", "me"]
        assert dataset[1].text2 == ["stand", "unfold", "yourself"]
        assert dataset[1].label == 0

        assert dataset[2].text1 == ["is", "Horatio"]
        assert dataset[2].text2 == ["a", "piece", "of", "him"]
        assert dataset[2].label == 0

        # Test filter remove all words in example
        dataset, text_field = filter_init(
            example1_values,
            example2_values,
            example3_values
        )

        text_field.vocab.stoi.pop("who")
        text_field.vocab.stoi.pop("is")
        text_field.vocab.stoi.pop("there")

        dataset.filter_examples(["text1", "text2"])

        assert dataset[0].text1 == []
        assert dataset[0].text2 == ["bernardo"]
        assert dataset[0].label == 1

        assert dataset[1].text1 == ["nay", "answer", "me"]
        assert dataset[1].text2 == ["stand", "unfold", "yourself"]
        assert dataset[1].label == 0

        assert dataset[2].text1 == ["Horatio"]
        assert dataset[2].text2 == ["a", "piece", "of", "him"]
        assert dataset[2].label == 0

    def test_gz_extraction(self):
        # tar.gz file contains train.txt and test.txt
        tgz = (b'\x1f\x8b\x08\x00\x1e\xcc\xd5Z\x00\x03\xed\xd1;\n\x800\x10E'
               b'\xd1,%+\x90\xc9G\xb3\x1e\x0b\x0b\x1b\x03q\x04\x97\xef\xa7'
               b'\xb0\xb0P,R\x08\xf74o`\x9aa\x9e\x96~\x9c\x1a]\xd5\xd4#\xbb'
               b'\x94\xd2\x99\xbb{\x9e\xb3\x0b\xbekC\x8c\x12\x9c\x11\xe7b\x10c'
               b'\xa5\xe2M\x97e\xd6\xbeXkJ\xce\x8f?x\xdb\xff\x94\x0e\xb3V\xae'
               b'\xff[\xffQ\x8e\xfe}\xf2\xf4\x0f\x00\x00\x00\x00\x00\x00\x00'
               b'\x00\x00\x00\x00\x00\x00O6\x1c\xc6\xbd\x89\x00(\x00\x00')

        # .gz file contains dummy.txt
        gz = (b'\x1f\x8b\x08\x08W\xce\xd5Z\x00\x03dummy.txt\x00\x0bq\r\x0e\x01'
              b'\x00\xb8\x93\xea\xee\x04\x00\x00\x00')

        # Create both files
        with open(os.path.join(self.test_dir, 'dummy.tar.gz'), 'wb') as fp:
            fp.write(tgz)

        with open(os.path.join(self.test_dir, 'dummy.txt.gz'), 'wb') as fp:
            fp.write(gz)

        # Set the urls in a dummy class
        class DummyDataset(data.Dataset):
            urls = ['dummy.tar.gz', 'dummy.txt.gz']
            name = ''
            dirname = ''

        # Run extraction
        DummyDataset.download(self.test_dir, check='')

        # Check if files were extracted correctly
        assert os.path.isfile(os.path.join(self.test_dir, 'dummy.txt'))
        assert os.path.isfile(os.path.join(self.test_dir, 'train.txt'))
        assert os.path.isfile(os.path.join(self.test_dir, 'test.txt'))


def filter_init(ex_val1, ex_val2, ex_val3):
    text_field = data.Field(sequential=True)
    label_field = data.Field(sequential=False)
    fields = [("text1", text_field), ("text2", text_field),
              ("label", label_field)]

    example1 = data.Example.fromlist(ex_val1, fields)
    example2 = data.Example.fromlist(ex_val2, fields)
    example3 = data.Example.fromlist(ex_val3, fields)
    examples = [example1, example2, example3]

    dataset = data.Dataset(examples, fields)
    text_field.build_vocab(dataset)

    return dataset, text_field
