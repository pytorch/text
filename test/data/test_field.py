# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter

from numpy.testing import assert_allclose
import torch
import torchtext.data as data
import pytest

from ..common.torchtext_test_case import TorchtextTestCase, verify_numericalized_example


class TestField(TorchtextTestCase):
    def test_process(self):
        raw_field = data.RawField()
        field = data.Field(sequential=True, use_vocab=False, batch_first=True)

        # Test tensor-like batch data which is accepted by both RawField and Field
        batch = [[1,2,3], [2,3,4]]
        batch_tensor = torch.LongTensor(batch)

        raw_field_processed = raw_field.process(batch)
        field_processed = field.process(batch, device=-1, train=False)

        assert raw_field_processed == batch
        assert field_processed.data.equal(batch_tensor)

        # Test non-tensor data which is only accepted by RawField
        any_obj = [object() for _ in range(5)]

        raw_field_processed = raw_field.process(any_obj)
        assert any_obj == raw_field_processed

        with pytest.raises(TypeError):
            field.process(any_obj)

    def test_preprocess(self):
        # Default case.
        field = data.Field()
        assert field.preprocess("Test string.") == ["Test", "string."]

        # Test that lowercase is properly applied.
        field_lower = data.Field(lower=True)
        assert field_lower.preprocess("Test string.") == ["test", "string."]

        # Test that custom preprocessing pipelines are properly applied.
        preprocess_pipeline = data.Pipeline(lambda x: x + "!")
        field_preprocessing = data.Field(preprocessing=preprocess_pipeline,
                                         lower=True)
        assert field_preprocessing.preprocess("Test string.") == ["test!", "string.!"]

        # Test that non-sequential data is properly handled.
        field_not_sequential = data.Field(sequential=False, lower=True,
                                          preprocessing=preprocess_pipeline)
        assert field_not_sequential.preprocess("Test string.") == "test string.!"

        # Non-regression test that we do not try to decode unicode strings to unicode
        field_not_sequential = data.Field(sequential=False, lower=True,
                                          preprocessing=preprocess_pipeline)
        assert field_not_sequential.preprocess("ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T") == "ᑌᑎiᑕoᗪᕮ_tᕮ᙭t!"

    def test_pad(self):
        # Default case.
        field = data.Field()
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [["a", "sentence", "of", "data", "."],
                                     ["yet", "another", "<pad>", "<pad>", "<pad>"],
                                     ["one", "last", "sent", "<pad>", "<pad>"]]
        expected_lengths = [5, 2, 3]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.Field(include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

        # Test fix_length properly truncates and pads.
        field = data.Field(fix_length=3)
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [["a", "sentence", "of"],
                                     ["yet", "another", "<pad>"],
                                     ["one", "last", "sent"]]
        expected_lengths = [3, 2, 3]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.Field(fix_length=3, include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

        # Test init_token is properly handled.
        field = data.Field(fix_length=4, init_token="<bos>")
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [["<bos>", "a", "sentence", "of"],
                                     ["<bos>", "yet", "another", "<pad>"],
                                     ["<bos>", "one", "last", "sent"]]
        expected_lengths = [4, 3, 4]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.Field(fix_length=4, init_token="<bos>", include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

        # Test init_token and eos_token are properly handled.
        field = data.Field(init_token="<bos>", eos_token="<eos>")
        minibatch = [["a", "sentence", "of", "data", "."],
                     ["yet", "another"],
                     ["one", "last", "sent"]]
        expected_padded_minibatch = [
            ["<bos>", "a", "sentence", "of", "data", ".", "<eos>"],
            ["<bos>", "yet", "another", "<eos>", "<pad>", "<pad>", "<pad>"],
            ["<bos>", "one", "last", "sent", "<eos>", "<pad>", "<pad>"]]
        expected_lengths = [7, 4, 5]
        assert field.pad(minibatch) == expected_padded_minibatch
        field = data.Field(init_token="<bos>", eos_token="<eos>", include_lengths=True)
        assert field.pad(minibatch) == (expected_padded_minibatch, expected_lengths)

        # Test that non-sequential data is properly handled.
        field = data.Field(init_token="<bos>", eos_token="<eos>", sequential=False)
        minibatch = [["contradiction"],
                     ["neutral"],
                     ["entailment"]]
        assert field.pad(minibatch) == minibatch
        field = data.Field(init_token="<bos>", eos_token="<eos>",
                           sequential=False, include_lengths=True)
        assert field.pad(minibatch) == minibatch

    def test_build_vocab(self):
        # Set up fields
        question_field = data.Field(sequential=True)
        label_field = data.Field(sequential=False)

        # Write TSV dataset and construct a Dataset
        self.write_test_ppid_dataset(data_format="tsv")
        tsv_fields = [("id", None), ("q1", question_field),
                      ("q2", question_field), ("label", label_field)]
        tsv_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="tsv",
            fields=tsv_fields)

        # Write JSON dataset and construct a Dataset
        self.write_test_ppid_dataset(data_format="json")
        json_fields = {"question1": ("q1", question_field),
                       "question2": ("q2", question_field),
                       "label": ("label", label_field)}
        json_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="json",
            fields=json_fields)

        # Test build_vocab default
        question_field.build_vocab(tsv_dataset, json_dataset)
        assert question_field.vocab.freqs == Counter(
            {'When': 4, 'do': 4, 'you': 4, 'use': 4, 'instead': 4,
             'of': 4, 'was': 4, 'Lincoln': 4, 'born?': 4, 'シ': 2,
             'し?': 2, 'Where': 2, 'What': 2, 'is': 2, '2+2': 2,
             '"&"': 2, '"and"?': 2, 'Which': 2, 'location': 2,
             'Abraham': 2, '2+2=?': 2})
        expected_stoi = {'<unk>': 0, '<pad>': 1, 'Lincoln': 2, 'When': 3,
                         'born?': 4, 'do': 5, 'instead': 6, 'of': 7,
                         'use': 8, 'was': 9, 'you': 10, '"&"': 11,
                         '"and"?': 12, '2+2': 13, '2+2=?': 14, 'Abraham': 15,
                         'What': 16, 'Where': 17, 'Which': 18, 'is': 19,
                         'location': 20, 'し?': 21, 'シ': 22}
        assert dict(question_field.vocab.stoi) == expected_stoi
        # Turn the stoi dictionary into an itos list
        expected_itos = [x[0] for x in sorted(expected_stoi.items(),
                                              key=lambda tup: tup[1])]
        assert question_field.vocab.itos == expected_itos

        label_field.build_vocab(tsv_dataset, json_dataset)
        assert label_field.vocab.freqs == Counter({'1': 4, '0': 2})
        expected_stoi = {'1': 1, '0': 2, '<unk>': 0}
        assert dict(label_field.vocab.stoi) == expected_stoi
        # Turn the stoi dictionary into an itos list
        expected_itos = [x[0] for x in sorted(expected_stoi.items(),
                                              key=lambda tup: tup[1])]
        assert label_field.vocab.itos == expected_itos

        # Test build_vocab default
        question_field.build_vocab(tsv_dataset, json_dataset)
        assert question_field.vocab.freqs == Counter(
            {'When': 4, 'do': 4, 'you': 4, 'use': 4, 'instead': 4,
             'of': 4, 'was': 4, 'Lincoln': 4, 'born?': 4, 'シ': 2,
             'し?': 2, 'Where': 2, 'What': 2, 'is': 2, '2+2': 2,
             '"&"': 2, '"and"?': 2, 'Which': 2, 'location': 2,
             'Abraham': 2, '2+2=?': 2})
        expected_stoi = {'<unk>': 0, '<pad>': 1, 'Lincoln': 2, 'When': 3,
                         'born?': 4, 'do': 5, 'instead': 6, 'of': 7,
                         'use': 8, 'was': 9, 'you': 10, '"&"': 11,
                         '"and"?': 12, '2+2': 13, '2+2=?': 14, 'Abraham': 15,
                         'What': 16, 'Where': 17, 'Which': 18, 'is': 19,
                         'location': 20, 'し?': 21, 'シ': 22}
        assert dict(question_field.vocab.stoi) == expected_stoi
        # Turn the stoi dictionary into an itos list
        expected_itos = [x[0] for x in sorted(expected_stoi.items(),
                                              key=lambda tup: tup[1])]
        assert question_field.vocab.itos == expected_itos

        label_field.build_vocab(tsv_dataset, json_dataset)
        assert label_field.vocab.freqs == Counter({'1': 4, '0': 2})
        expected_stoi = {'1': 1, '0': 2, '<unk>': 0}
        assert dict(label_field.vocab.stoi) == expected_stoi
        # Turn the stoi dictionary into an itos list
        expected_itos = [x[0] for x in sorted(expected_stoi.items(),
                                              key=lambda tup: tup[1])]
        assert label_field.vocab.itos == expected_itos

        # Test build_vocab with extra kwargs passed to Vocab
        question_field.build_vocab(tsv_dataset, json_dataset, max_size=8,
                                   min_freq=3)
        assert question_field.vocab.freqs == Counter(
            {'When': 4, 'do': 4, 'you': 4, 'use': 4, 'instead': 4,
             'of': 4, 'was': 4, 'Lincoln': 4, 'born?': 4, 'シ': 2,
             'し?': 2, 'Where': 2, 'What': 2, 'is': 2, '2+2': 2,
             '"&"': 2, '"and"?': 2, 'Which': 2, 'location': 2,
             'Abraham': 2, '2+2=?': 2})
        expected_stoi = {'<unk>': 0, '<pad>': 1, 'Lincoln': 2, 'When': 3,
                         'born?': 4, 'do': 5, 'instead': 6, 'of': 7,
                         'use': 8, 'was': 9}
        assert dict(question_field.vocab.stoi) == expected_stoi
        # Turn the stoi dictionary into an itos list
        expected_itos = [x[0] for x in sorted(expected_stoi.items(),
                                              key=lambda tup: tup[1])]
        assert question_field.vocab.itos == expected_itos

    def test_numericalize_basic(self):
        self.write_test_ppid_dataset(data_format="tsv")
        question_field = data.Field(sequential=True)
        tsv_fields = [("id", None), ("q1", question_field),
                      ("q2", question_field), ("label", None)]
        tsv_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="tsv",
            fields=tsv_fields)
        question_field.build_vocab(tsv_dataset)

        test_example_data = [["When", "do", "you", "use", "シ",
                              "instead", "of", "し?"],
                             ["What", "is", "2+2", "<pad>", "<pad>",
                              "<pad>", "<pad>", "<pad>"],
                             ["Here", "is", "a", "sentence", "with",
                              "some", "oovs", "<pad>"]]

        # Test default
        default_numericalized = question_field.numericalize(
            test_example_data, device=-1)
        verify_numericalized_example(question_field, test_example_data,
                                     default_numericalized)
        # Test with train=False
        volatile_numericalized = question_field.numericalize(
            test_example_data, device=-1, train=False)
        verify_numericalized_example(question_field, test_example_data,
                                     volatile_numericalized, train=False)

    def test_numericalize_include_lengths(self):
        self.write_test_ppid_dataset(data_format="tsv")
        question_field = data.Field(sequential=True, include_lengths=True)
        tsv_fields = [("id", None), ("q1", question_field),
                      ("q2", question_field), ("label", None)]
        tsv_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="tsv",
            fields=tsv_fields)
        question_field.build_vocab(tsv_dataset)

        test_example_data = [["When", "do", "you", "use", "シ",
                              "instead", "of", "し?"],
                             ["What", "is", "2+2", "<pad>", "<pad>",
                              "<pad>", "<pad>", "<pad>"],
                             ["Here", "is", "a", "sentence", "with",
                              "some", "oovs", "<pad>"]]
        test_example_lengths = [8, 3, 7]

        # Test with include_lengths
        include_lengths_numericalized = question_field.numericalize(
            (test_example_data, test_example_lengths), device=-1)
        verify_numericalized_example(question_field,
                                     test_example_data,
                                     include_lengths_numericalized,
                                     test_example_lengths)

    def test_numericalize_batch_first(self):
        self.write_test_ppid_dataset(data_format="tsv")
        question_field = data.Field(sequential=True, batch_first=True)
        tsv_fields = [("id", None), ("q1", question_field),
                      ("q2", question_field), ("label", None)]
        tsv_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="tsv",
            fields=tsv_fields)
        question_field.build_vocab(tsv_dataset)

        test_example_data = [["When", "do", "you", "use", "シ",
                              "instead", "of", "し?"],
                             ["What", "is", "2+2", "<pad>", "<pad>",
                              "<pad>", "<pad>", "<pad>"],
                             ["Here", "is", "a", "sentence", "with",
                              "some", "oovs", "<pad>"]]

        # Test with batch_first
        include_lengths_numericalized = question_field.numericalize(
            test_example_data, device=-1)
        verify_numericalized_example(question_field,
                                     test_example_data,
                                     include_lengths_numericalized,
                                     batch_first=True)

    def test_numericalize_postprocessing(self):
        self.write_test_ppid_dataset(data_format="tsv")

        def reverse_postprocess(arr, vocab, train):
            return [list(reversed(sentence)) for sentence in arr]

        question_field = data.Field(sequential=True,
                                    postprocessing=reverse_postprocess)
        tsv_fields = [("id", None), ("q1", question_field),
                      ("q2", question_field), ("label", None)]

        tsv_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="tsv",
            fields=tsv_fields)
        question_field.build_vocab(tsv_dataset)

        test_example_data = [["When", "do", "you", "use", "シ",
                              "instead", "of", "し?"],
                             ["What", "is", "2+2", "<pad>", "<pad>",
                              "<pad>", "<pad>", "<pad>"],
                             ["Here", "is", "a", "sentence", "with",
                              "some", "oovs", "<pad>"]]
        reversed_test_example_data = [list(reversed(sentence)) for sentence in
                                      test_example_data]

        postprocessed_numericalized = question_field.numericalize(
            (test_example_data), device=-1)
        verify_numericalized_example(question_field,
                                     reversed_test_example_data,
                                     postprocessed_numericalized)

    def test_numerical_features_no_vocab(self):
        self.write_test_numerical_features_dataset()
        # Test basic usage
        int_field = data.Field(sequential=False, use_vocab=False)
        float_field = data.Field(sequential=False, use_vocab=False,
                                 tensor_type=torch.FloatTensor)
        tsv_fields = [("int", int_field), ("float", float_field), ("string", None)]
        tsv_dataset = data.TabularDataset(
            path=self.test_numerical_features_dataset_path, format="tsv",
            fields=tsv_fields)
        int_field.build_vocab(tsv_dataset)
        float_field.build_vocab(tsv_dataset)
        test_int_data = ["1", "0", "1", "3", "19"]
        test_float_data = ["1.1", "0.1", "3.91", "0.2", "10.2"]

        numericalized_int = int_field.numericalize(test_int_data, device=-1)
        assert_allclose(numericalized_int.data.numpy(), [1, 0, 1, 3, 19])
        numericalized_float = float_field.numericalize(test_float_data, device=-1)
        assert_allclose(numericalized_float.data.numpy(), [1.1, 0.1, 3.91, 0.2, 10.2])

        # Test with postprocessing applied
        int_field = data.Field(sequential=False, use_vocab=False,
                               postprocessing=lambda arr, _, __: [x + 1 for x in arr])
        float_field = data.Field(sequential=False, use_vocab=False,
                                 tensor_type=torch.FloatTensor,
                                 postprocessing=lambda arr, _, __: [x * 0.5 for x in arr])
        tsv_fields = [("int", int_field), ("float", float_field), ("string", None)]
        tsv_dataset = data.TabularDataset(
            path=self.test_numerical_features_dataset_path, format="tsv",
            fields=tsv_fields)
        int_field.build_vocab(tsv_dataset)
        float_field.build_vocab(tsv_dataset)
        test_int_data = ["1", "0", "1", "3", "19"]
        test_float_data = ["1.1", "0.1", "3.91", "0.2", "10.2"]

        numericalized_int = int_field.numericalize(test_int_data, device=-1)
        assert_allclose(numericalized_int.data.numpy(), [2, 1, 2, 4, 20])
        numericalized_float = float_field.numericalize(test_float_data, device=-1)
        assert_allclose(numericalized_float.data.numpy(), [0.55, 0.05, 1.955, 0.1, 5.1])

    def test_errors(self):
        # Test that passing a non-tuple (of data and length) to numericalize
        # with Field.include_lengths = True raises an error.
        with self.assertRaises(ValueError):
            self.write_test_ppid_dataset(data_format="tsv")
            question_field = data.Field(sequential=True, include_lengths=True)
            tsv_fields = [("id", None), ("q1", question_field),
                          ("q2", question_field), ("label", None)]
            tsv_dataset = data.TabularDataset(
                path=self.test_ppid_dataset_path, format="tsv",
                fields=tsv_fields)
            question_field.build_vocab(tsv_dataset)
            test_example_data = [["When", "do", "you", "use", "シ",
                                  "instead", "of", "し?"],
                                 ["What", "is", "2+2", "<pad>", "<pad>",
                                  "<pad>", "<pad>", "<pad>"],
                                 ["Here", "is", "a", "sentence", "with",
                                  "some", "oovs", "<pad>"]]
            question_field.numericalize(
                test_example_data, device=-1)
