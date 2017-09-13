# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter

import torchtext.data as data

from ..common.torchtext_test_case import TorchtextTestCase


class TestField(TorchtextTestCase):
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
        test_example_numericalized.data.t_()
    # Transpose numericalized example so we can compare over batches
    for example_idx, numericalized_single_example in enumerate(
            test_example_numericalized.t()):
        assert len(test_example_data[example_idx]) == len(numericalized_single_example)
        assert numericalized_single_example.volatile is not train
        for token_idx, numericalized_token in enumerate(
                numericalized_single_example):
            # Convert from Variable to int
            numericalized_token = numericalized_token.data[0]
            test_example_token = test_example_data[example_idx][token_idx]
            # Check if the numericalized example is correct, taking into
            # account unknown tokens.
            if field.vocab.stoi[test_example_token] != 0:
                # token is in-vocabulary
                assert (field.vocab.itos[numericalized_token] ==
                        test_example_token)
            else:
                # token is OOV and <unk> always has an index of 0
                assert numericalized_token == 0
