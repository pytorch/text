from torchtext.data import get_tokenizer

from ..common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):
    TEST_STR = "A string, particularly one with slightly complex punctuation."

    def test_get_tokenizer_split(self):
        # Test the default case with str.split
        assert get_tokenizer(str.split) == str.split
        assert get_tokenizer(str.split)(self.TEST_STR) == str.split(self.TEST_STR)

    def test_get_tokenizer_toktokt(self):
        # Test Toktok option. Test strings taken from NLTK doctests.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        toktok_tokenizer = get_tokenizer("toktok")
        assert toktok_tokenizer(self.TEST_STR) == [
            "A",
            "string",
            ",",
            "particularly",
            "one",
            "with",
            "slightly",
            "complex",
            "punctuation",
            ".",
        ]

        # Test that errors are raised for invalid input arguments.
        with self.assertRaises(ValueError):
            get_tokenizer(1)
        with self.assertRaises(ValueError):
            get_tokenizer("some other string")
