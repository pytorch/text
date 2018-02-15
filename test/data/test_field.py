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
        batch = [[1, 2, 3], [2, 3, 4]]
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


class TestNestedField(TorchtextTestCase):
    def test_init_minimal(self):
        nesting_field = data.Field()
        field = data.NestedField(nesting_field)

        assert isinstance(field, data.Field)
        assert field.nesting_field is nesting_field
        assert field.sequential
        assert field.use_vocab
        assert field.init_token is None
        assert field.eos_token is None
        assert field.unk_token == nesting_field.unk_token
        assert field.fix_length is None
        assert field.tensor_type is torch.LongTensor
        assert field.preprocessing is None
        assert field.postprocessing is None
        assert field.lower == nesting_field.lower
        assert field.tokenize("a b c") == "a b c".split()
        assert not field.include_lengths
        assert field.batch_first
        assert field.pad_token == nesting_field.pad_token
        assert not field.pad_first

    def test_init_when_nesting_field_is_not_sequential(self):
        nesting_field = data.Field(sequential=False)
        field = data.NestedField(nesting_field)

        assert field.pad_token == "<pad>"

    def test_init_when_nesting_field_has_include_lengths_equal_true(self):
        nesting_field = data.Field(include_lengths=True)

        with pytest.raises(ValueError) as excinfo:
            data.NestedField(nesting_field)
        assert "nesting field cannot have include_lengths=True" in str(excinfo.value)

    def test_init_with_nested_field_as_nesting_field(self):
        nesting_field = data.NestedField(data.Field())

        with pytest.raises(ValueError) as excinfo:
            data.NestedField(nesting_field)
        assert "nesting field must not be another NestedField" in str(excinfo.value)

    def test_init_full(self):
        nesting_field = data.Field()
        field = data.NestedField(
            nesting_field,
            use_vocab=False,
            init_token="<s>",
            eos_token="</s>",
            fix_length=10,
            tensor_type=torch.FloatTensor,
            preprocessing=lambda xs: list(reversed(xs)),
            postprocessing=lambda xs: [x.upper() for x in xs],
            tokenize=list,
            pad_first=True,
        )

        assert not field.use_vocab
        assert field.init_token == "<s>"
        assert field.eos_token == "</s>"
        assert field.fix_length == 10
        assert field.tensor_type is torch.FloatTensor
        assert field.preprocessing("a b c".split()) == "c b a".split()
        assert field.postprocessing("a b c".split()) == "A B C".split()
        assert field.tokenize("abc") == ["a", "b", "c"]
        assert field.pad_first

    def test_preprocess(self):
        nesting_field = data.Field(
            tokenize=list, preprocessing=lambda xs: [x.upper() for x in xs])
        field = data.NestedField(nesting_field, preprocessing=lambda xs: reversed(xs))
        preprocessed = field.preprocess("john loves mary")

        assert preprocessed == [list("MARY"), list("LOVES"), list("JOHN")]

    def test_build_vocab_from_dataset(self):
        nesting_field = data.Field(tokenize=list, unk_token="<cunk>", pad_token="<cpad>",
                                   init_token="<w>", eos_token="</w>")
        CHARS = data.NestedField(nesting_field, init_token="<s>", eos_token="</s>")
        ex1 = data.Example.fromlist(["aaa bbb c"], [("chars", CHARS)])
        ex2 = data.Example.fromlist(["bbb aaa"], [("chars", CHARS)])
        dataset = data.Dataset([ex1, ex2], [("chars", CHARS)])

        CHARS.build_vocab(dataset, min_freq=2)

        expected = "a b <w> </w> <s> </s> <cunk> <cpad>".split()
        assert len(CHARS.vocab) == len(expected)
        for c in expected:
            assert c in CHARS.vocab.stoi

    def test_build_vocab_from_iterable(self):
        nesting_field = data.Field(unk_token="<cunk>", pad_token="<cpad>")
        CHARS = data.NestedField(nesting_field)
        CHARS.build_vocab(
            [[list("aaa"), list("bbb"), ["c"]], [list("bbb"), list("aaa")]],
            [[list("ccc"), list("bbb")], [list("bbb")]],
        )

        expected = "a b c <cunk> <cpad>".split()
        assert len(CHARS.vocab) == len(expected)
        for c in expected:
            assert c in CHARS.vocab.stoi

    def test_pad(self):
        nesting_field = data.Field(tokenize=list, unk_token="<cunk>", pad_token="<cpad>",
                                   init_token="<w>", eos_token="</w>")
        CHARS = data.NestedField(nesting_field, init_token="<s>", eos_token="</s>")
        minibatch = [
            [list("john"), list("loves"), list("mary")],
            [list("mary"), list("cries")],
        ]
        expected = [
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("john") + ["</w>", "<cpad>"],
                ["<w>"] + list("loves") + ["</w>"],
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
            ],
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>"] + list("cries") + ["</w>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
                ["<cpad>"] * 7,
            ]
        ]

        assert CHARS.pad(minibatch) == expected

    def test_pad_when_nesting_field_is_not_sequential(self):
        nesting_field = data.Field(sequential=False, unk_token="<cunk>",
                                   pad_token="<cpad>", init_token="<w>", eos_token="</w>")
        CHARS = data.NestedField(nesting_field, init_token="<s>", eos_token="</s>")
        minibatch = [
            ["john", "loves", "mary"],
            ["mary", "cries"]
        ]
        expected = [
            ["<s>", "john", "loves", "mary", "</s>"],
            ["<s>", "mary", "cries", "</s>", "<pad>"],
        ]

        assert CHARS.pad(minibatch) == expected

    def test_pad_when_nesting_field_has_fix_length(self):
        nesting_field = data.Field(tokenize=list, unk_token="<cunk>", pad_token="<cpad>",
                                   init_token="<w>", eos_token="</w>", fix_length=5)
        CHARS = data.NestedField(nesting_field, init_token="<s>", eos_token="</s>")
        minibatch = [
            ["john", "loves", "mary"],
            ["mary", "cries"]
        ]
        expected = [
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 2,
                ["<w>"] + list("joh") + ["</w>"],
                ["<w>"] + list("lov") + ["</w>"],
                ["<w>"] + list("mar") + ["</w>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 2,
            ],
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 2,
                ["<w>"] + list("mar") + ["</w>"],
                ["<w>"] + list("cri") + ["</w>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 2,
                ["<cpad>"] * 5,
            ]
        ]

        assert CHARS.pad(minibatch) == expected

    def test_pad_when_fix_length_is_not_none(self):
        nesting_field = data.Field(tokenize=list, unk_token="<cunk>", pad_token="<cpad>",
                                   init_token="<w>", eos_token="</w>")
        CHARS = data.NestedField(
            nesting_field, init_token="<s>", eos_token="</s>", fix_length=3)
        minibatch = [
            ["john", "loves", "mary"],
            ["mary", "cries"]
        ]
        expected = [
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("john") + ["</w>", "<cpad>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
            ],
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
            ]
        ]

        assert CHARS.pad(minibatch) == expected

    def test_pad_when_no_init_and_eos_tokens(self):
        nesting_field = data.Field(tokenize=list, unk_token="<cunk>", pad_token="<cpad>",
                                   init_token="<w>", eos_token="</w>")
        CHARS = data.NestedField(nesting_field)
        minibatch = [
            ["john", "loves", "mary"],
            ["mary", "cries"]
        ]
        expected = [
            [
                ["<w>"] + list("john") + ["</w>", "<cpad>"],
                ["<w>"] + list("loves") + ["</w>"],
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
            ],
            [
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>"] + list("cries") + ["</w>"],
                ["<cpad>"] * 7,
            ]
        ]

        assert CHARS.pad(minibatch) == expected

    def test_pad_when_pad_first_is_true(self):
        nesting_field = data.Field(tokenize=list, unk_token="<cunk>", pad_token="<cpad>",
                                   init_token="<w>", eos_token="</w>")
        CHARS = data.NestedField(nesting_field, init_token="<s>", eos_token="</s>",
                                 pad_first=True)
        minibatch = [
            [list("john"), list("loves"), list("mary")],
            [list("mary"), list("cries")],
        ]
        expected = [
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("john") + ["</w>", "<cpad>"],
                ["<w>"] + list("loves") + ["</w>"],
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
            ],
            [
                ["<cpad>"] * 7,
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>"] + list("cries") + ["</w>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
            ]
        ]

        assert CHARS.pad(minibatch) == expected

    def test_numericalize(self):
        nesting_field = data.Field(batch_first=True)
        field = data.NestedField(nesting_field)
        ex1 = data.Example.fromlist(["john loves mary"], [("words", field)])
        ex2 = data.Example.fromlist(["mary cries"], [("words", field)])
        dataset = data.Dataset([ex1, ex2], [("words", field)])
        field.build_vocab(dataset)
        examples_data = [
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("john") + ["</w>", "<cpad>"],
                ["<w>"] + list("loves") + ["</w>"],
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
            ],
            [
                ["<w>", "<s>", "</w>"] + ["<cpad>"] * 4,
                ["<w>"] + list("mary") + ["</w>", "<cpad>"],
                ["<w>"] + list("cries") + ["</w>"],
                ["<w>", "</s>", "</w>"] + ["<cpad>"] * 4,
                ["<cpad>"] * 7,
            ]
        ]
        numericalized = field.numericalize(examples_data, device=-1)

        assert numericalized.dim() == 3
        assert numericalized.size(0) == len(examples_data)
        for example, numericalized_example in zip(examples_data, numericalized):
            verify_numericalized_example(
                field, example, numericalized_example, batch_first=True)


class TestLabelField(TorchtextTestCase):
    def test_init(self):
        # basic init
        label_field = data.LabelField()
        assert label_field.sequential is False
        assert label_field.unk_token is None

        # init with preset fields
        label_field = data.LabelField(sequential=True, unk_token="<unk>")
        assert label_field.sequential is False
        assert label_field.unk_token is None

    def test_vocab_size(self):
        # Set up fields
        question_field = data.Field(sequential=True)
        label_field = data.LabelField()

        # Copied from test_build_vocab with minor changes
        # Write TSV dataset and construct a Dataset
        self.write_test_ppid_dataset(data_format="tsv")
        tsv_fields = [("id", None), ("q1", question_field),
                      ("q2", question_field), ("label", label_field)]
        tsv_dataset = data.TabularDataset(
            path=self.test_ppid_dataset_path, format="tsv",
            fields=tsv_fields)

        # Skipping json dataset as we can rely on the original build vocab test
        label_field.build_vocab(tsv_dataset)
        assert label_field.vocab.freqs == Counter({'1': 2, '0': 1})
        expected_stoi = {'1': 0, '0': 1}  # No <unk>
        assert dict(label_field.vocab.stoi) == expected_stoi
        # Turn the stoi dictionary into an itos list
        expected_itos = [x[0] for x in sorted(expected_stoi.items(),
                                              key=lambda tup: tup[1])]
        assert label_field.vocab.itos == expected_itos
