import six
import torchtext.data as data
import pytest
from ..common.torchtext_test_case import TorchtextTestCase
from torchtext.datasets.text_classification import text_normalize


class TestUtils(TorchtextTestCase):
    TEST_STR = "A string, particularly one with slightly complex punctuation."

    def test_get_tokenizer_split(self):
        # Test the default case with str.split
        assert data.get_tokenizer(str.split) == str.split
        assert data.get_tokenizer(str.split)(self.TEST_STR) == str.split(self.TEST_STR)

    def test_get_tokenizer_spacy(self):
        # Test SpaCy option, and verify it properly handles punctuation.
        assert data.get_tokenizer("spacy")(six.text_type(self.TEST_STR)) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

    # TODO: Remove this once issue was been resolved.
    @pytest.mark.skip(reason=("Impractically slow! "
                              "https://github.com/alvations/sacremoses/issues/61"))
    def test_get_tokenizer_moses(self):
        # Test Moses option.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        moses_tokenizer = data.get_tokenizer("moses")
        assert moses_tokenizer(self.TEST_STR) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Nonbreaking prefixes should tokenize the final period.
        assert moses_tokenizer(six.text_type("abc def.")) == ["abc", "def", "."]

    def test_get_tokenizer_toktokt(self):
        # Test Toktok option. Test strings taken from NLTK doctests.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        toktok_tokenizer = data.get_tokenizer("toktok")
        assert toktok_tokenizer(self.TEST_STR) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Test that errors are raised for invalid input arguments.
        with self.assertRaises(ValueError):
            data.get_tokenizer(1)
        with self.assertRaises(ValueError):
            data.get_tokenizer("some other string")

    def test_text_nomalize_function(self):
        # Test text_nomalize function in torchtext.datasets.text_classification
        origin_line = r'U.S. Stocks Set for Slightly Firmer Open"," NEW YORK (Reuters) - U.S. stocks were set to move slightly  higher at the open on Friday with a \$1 billion share buyback at Texas Instruments Inc. &lt;A HREF=""http://www.investor.reuters.com/FullQuote.aspx?ticker=TXN.N target=/stocks/quickinfo/fullquote""&gt;TXN.N&lt;/A&gt; supporting technology stocks,  although rising oil prices could limit gains.'
        ref_result = r'u . s . stocks set for slightly firmer open , new york ( reuters ) - u . s . stocks were set to move slightly higher at the open on friday with a \$1 billion share buyback at texas instruments inc . &lt a href=http //www . investor . reuters . com/fullquote . aspx ? ticker=txn . n target=/stocks/quickinfo/fullquote&gt txn . n&lt /a&gt supporting technology stocks , although rising oil prices could limit gains .'

        test_result = text_normalize(origin_line)
        assert test_result == ref_result.split()
