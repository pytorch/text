import io
from torchtext.data import get_tokenizer
from torchtext.utils import unicode_csv_reader
from torchtext.experimental.functional import ngrams_func
from ..common.torchtext_test_case import TorchtextTestCase
from ..common.assets import get_asset_path


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
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Test that errors are raised for invalid input arguments.
        with self.assertRaises(ValueError):
            get_tokenizer(1)
        with self.assertRaises(ValueError):
            get_tokenizer("some other string")

    def test_text_nomalize_function(self):
        # Test text_nomalize function in torchtext.datasets.text_classification
        ref_lines = []
        test_lines = []

        tokenizer = get_tokenizer("basic_english")
        data_path = get_asset_path('text_normalization_ag_news_test.csv')
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                test_lines.append(tokenizer(' , '.join(row)))

        data_path = get_asset_path('text_normalization_ag_news_ref_results.test')
        with io.open(data_path, encoding="utf8") as ref_data:
            for line in ref_data:
                line = line.split()
                self.assertEqual(line[0][:9], '__label__')
                line[0] = line[0][9:]  # remove '__label__'
                ref_lines.append(line)

        self.assertEqual(ref_lines, test_lines)

    def test_ngrams_func(self):
        func = ngrams_func(1)
        assert func(['A', 'string', 'particularly', 'one', 'with', 'slightly']) == \
            ['A', 'string', 'particularly', 'one', 'with', 'slightly']
        func = ngrams_func(2)
        assert func(['A', 'string', 'particularly', 'one', 'with', 'slightly']) == \
            ['A', 'string', 'particularly', 'one', 'with', 'slightly', 'A string', 'string particularly',
             'particularly one', 'one with', 'with slightly']
        func = ngrams_func(3)
        assert func(['A', 'string', 'particularly', 'one', 'with', 'slightly']) == \
            ['A', 'string', 'particularly', 'one', 'with', 'slightly', 'A string', 'string particularly',
             'particularly one', 'one with', 'with slightly', 'A string particularly',
             'string particularly one', 'particularly one with', 'one with slightly']
