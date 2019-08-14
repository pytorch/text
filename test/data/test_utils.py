import six
import torchtext.data as data
import pytest
from ..common.torchtext_test_case import TorchtextTestCase
from torchtext.utils import unicode_csv_reader
import io


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
        import sys, os
        def trace(frame, event, arg):
            print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
            return trace
        
        print('asdf')
        # ref_lines = []
        # test_lines = []
        import _C
        print(_C.__dir__())
        print(_C.x)
        print(_C.__file__)
        print(os.path.exists(_C.__file__))
        sys.settrace(trace)
        _C.x(121)
        print('done')
        # assert _C.x(121) == 122, 'rekee: %d' % _C.x(121)
        # tokenizer = data.get_tokenizer("basic_english")

        # data_path = 'test/asset/text_normalization_ag_news_test.csv'
        # with io.open(data_path, encoding="utf8") as f:
        #     reader = unicode_csv_reader(f)
        #     for row in reader:
        #         y = tokenizer(121)
        #         assert y == 122, "rek: %d" % (y)
                # raise Exception(tokenizer('jason'))
                # test_lines.append(tokenizer(' , '.join(row)))
                # assert test_lines[-1] == 'hello world'
        # raise Exception(test_lines)

        # data_path = 'test/asset/text_normalization_ag_news_ref_results.test'
        # with io.open(data_path, encoding="utf8") as ref_data:
        #     for line in ref_data:
        #         line = line.split()
        #         self.assertEqual(line[0][:9], '__label__')
        #         line[0] = line[0][9:]  # remove '__label__'
        #         ref_lines.append(line)

        # self.assertEqual(ref_lines, test_lines)

