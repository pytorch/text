from ..common.torchtext_test_case import TorchtextTestCase
import sentencepiece as spm


class TestUtils(TorchtextTestCase):
    def test_sentencepiece_encode_as_ids(self):
        from torchtext.data.functional import sentencepiece_encode_as_ids
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(model_path)

        ref_results = [15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                       144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]

        self.assertEqual(sentencepiece_encode_as_ids(sp_model, test_sample),
                         ref_results)

    def test_sentencepiece_encode_as_pieces(self):
        import sys
        from torchtext.data.functional import sentencepiece_encode_as_pieces

        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(model_path)

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

        self.assertEqual(sentencepiece_encode_as_pieces(sp_model, test_sample),
                         ref_results)

    def test_generate_sp_tokenizer(self):
        # Test the function to train a sentencepiece tokenizer
        from torchtext.data.functional import generate_sp_tokenizer
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
