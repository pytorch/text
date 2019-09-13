import six
import torchtext.data as data
import pytest
from ..common.torchtext_test_case import TorchtextTestCase
from torchtext.utils import unicode_csv_reader
import io
import sys


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
    # TODO# Add nltk data back in build_tools/travis/install.sh.
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
        ref_lines = []
        test_lines = []

        tokenizer = data.get_tokenizer("basic_english")
        data_path = 'test/asset/text_normalization_ag_news_test.csv'
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                test_lines.append(tokenizer(' , '.join(row)))

        data_path = 'test/asset/text_normalization_ag_news_ref_results.test'
        with io.open(data_path, encoding="utf8") as ref_data:
            for line in ref_data:
                line = line.split()
                self.assertEqual(line[0][:9], '__label__')
                line[0] = line[0][9:]  # remove '__label__'
                ref_lines.append(line)

        self.assertEqual(ref_lines, test_lines)

    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason="SyntaxError: Non-ASCII character in python 2.7")
    def test_get_tokenizer_sentencepiece(self):
        # Test SentencePiece option, and verify it properly handles punctuation.
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        tokenizer = data.get_tokenizer("sentencepiece", spm_name=model_path)

        ref_results = ['▁Sent', 'ence', 'P', 'ie', 'ce', '▁is', '▁an', '▁un',
                       'super', 'vis', 'ed', '▁text', '▁to', 'ken', 'izer', '▁and',
                       '▁de', 'to', 'ken', 'izer']

        self.assertEqual(tokenizer(test_sample), ref_results)

    def test_generate_sp_tokenizer(self):
        # Test the function to train a sentencepiece tokenizer
        from torchtext.data.utils import generate_sp_tokenizer
        import sentencepiece as spm
        import os

        data_path = 'test/asset/text_normalization_ag_news_test.csv'
        generate_sp_tokenizer(data_path,
                              vocab_size=23456,
                              model_prefix='spm_user')

        sp_user = spm.SentencePieceProcessor()
        sp_user.Load('spm_user.model')

        self.assertEqual(len(sp_user), 23456)

        if os.path.isfile('spm_user.model'):
            os.remove('spm_user.model')
        if os.path.isfile('spm_user.vocab'):
            os.remove('spm_user.vocab')

    def test_spm_data_generator(self):
        # Test the function to generate data with sentencepiece tokenizer
        from torchtext.data.utils import spm_data_generator

        iterator = ['Generic data loaders, abstractions, and iterators',
                    'Pre-built loaders for common NLP datasets']
        model_path = 'test/asset/spm_example.model'

        results = spm_data_generator(model_path, iterator)

        self.assertEqual(results, [[15122, 6030, 13208, 4503, 755, 5, 7640, 9383,
                                    4703, 13, 5, 6, 15, 298, 4105, 13],
                                   [5212, 47, 3106, 1782, 20, 4503, 755, 18,
                                    2578, 1463, 1524, 981, 13208, 5116, 13]])
