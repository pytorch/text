import six
import torchtext.data as data

from ..common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):
    def test_get_tokenizer(self):
        # Test the default case with str.split
        assert data.get_tokenizer(str.split) == str.split
        test_str = "A string, particularly one with slightly complex punctuation."
        assert data.get_tokenizer(str.split)(test_str) == str.split(test_str)

        # Test SpaCy option, and verify it properly handles punctuation.
        assert data.get_tokenizer("spacy")(six.text_type(test_str)) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Test Moses option.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        moses_tokenizer = data.get_tokenizer("moses")
        assert moses_tokenizer(test_str) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Nonbreaking prefixes should tokenize the final period.
        assert moses_tokenizer(six.text_type("abc def.")) == ["abc", "def", "."]

        # Test Toktok option. Test strings taken from NLTK doctests.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        toktok_tokenizer = data.get_tokenizer("toktok")
        assert toktok_tokenizer(test_str) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Test that errors are raised for invalid input arguments.
        with self.assertRaises(ValueError):
            data.get_tokenizer(1)
        with self.assertRaises(ValueError):
            data.get_tokenizer("some other string")
