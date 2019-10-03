from ..common.torchtext_test_case import TorchtextTestCase
import sentencepiece as spm
from torchtext.data.functional import generate_sp_model, load_sp_model, \
    sentencepiece_numericalizer, sentencepiece_tokenizer, \
    custom_replace, simple_space_split
import os
import sys


class TestFunctional(TorchtextTestCase):
    def test_generate_sp_model(self):
        # Test the function to train a sentencepiece tokenizer

        data_path = 'test/asset/text_normalization_ag_news_test.csv'
        generate_sp_model(data_path,
                          vocab_size=23456,
                          model_prefix='spm_user')

        sp_user = spm.SentencePieceProcessor()
        sp_user.Load('spm_user.model')

        self.assertEqual(len(sp_user), 23456)

        if os.path.isfile('spm_user.model'):
            os.remove('spm_user.model')
        if os.path.isfile('spm_user.vocab'):
            os.remove('spm_user.vocab')

    def test_sentencepiece_numericalizer(self):
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_model = load_sp_model(model_path)
        self.assertEqual(len(sp_model), 20000)
        spm_generator = sentencepiece_numericalizer(sp_model)

        ref_results = [15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                       144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

    def test_sentencepiece_tokenizer(self):

        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_model = load_sp_model(model_path)
        self.assertEqual(len(sp_model), 20000)
        spm_generator = sentencepiece_tokenizer(sp_model)

        # Handle byte string in Python2 and Unicode string in Python3, respectively
        if sys.version_info < (3, 0):
            ref_results = ['\xe2\x96\x81Sent', 'ence', 'P', 'ie', 'ce', '\xe2\x96\x81is',
                           '\xe2\x96\x81an', '\xe2\x96\x81un', 'super', 'vis', 'ed',
                           '\xe2\x96\x81text', '\xe2\x96\x81to', 'ken', 'izer',
                           '\xe2\x96\x81and', '\xe2\x96\x81de', 'to', 'ken', 'izer']
        else:
            ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                           '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                           '\u2581to', 'ken', 'izer', '\u2581and',
                           '\u2581de', 'to', 'ken', 'izer']

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

    def test_custom_replace(self):
        custom_replace_transform = custom_replace([(r'S', 's'), (r'\s+', ' ')])
        test_sample = ['test     cuStom   replace', 'with   uSer   instruction']
        ref_results = ['test custom replace', 'with user instruction']

        self.assertEqual(list(custom_replace_transform(test_sample)),
                         ref_results)

    def test_simple_space_split(self):
        test_sample = ['test simple space split function']
        ref_results = ['test', 'simple', 'space', 'split', 'function']

        self.assertEqual(list(simple_space_split(test_sample))[0],
                         ref_results)
