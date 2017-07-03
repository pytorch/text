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
