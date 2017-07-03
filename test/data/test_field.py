from unittest import TestCase
import torchtext.data as data


class TestField(TestCase):
    def test_preprocess(self):
        field = data.Field()
        assert field.preprocess("Test string.") == ["Test", "string."]

        field_lower = data.Field(lower=True)
        assert field_lower.preprocess("Test string.") == ["test", "string."]

        preprocess_pipeline = data.Pipeline(lambda x: x + "!")
        field_preprocessing = data.Field(preprocessing=preprocess_pipeline,
                                         lower=True)
        assert field_preprocessing.preprocess("Test string.") == ["test!", "string.!"]

        field_not_sequential = data.Field(sequential=False, lower=True,
                                          preprocessing=preprocess_pipeline)
        assert field_not_sequential.preprocess("Test string.") == "test string.!"

    def test_pad(self):
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

        field = data.Field(init_token="<bos>", eos_token="<eos>", sequential=False)
        minibatch = [["contradiction"],
                     ["neutral"],
                     ["entailment"]]
        assert field.pad(minibatch) == minibatch
        field = data.Field(init_token="<bos>", eos_token="<eos>",
                           sequential=False, include_lengths=True)
        assert field.pad(minibatch) == minibatch
