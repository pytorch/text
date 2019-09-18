from ..common.torchtext_test_case import TorchtextTestCase
from torchtext.data.transforms import SentencePieceTransform


class TestUtils(TorchtextTestCase):
    def test_sentencepiece_transform(self):
        # sentencepiece transform.
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_transform = SentencePieceTransform(model_path)

        import sys
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

        self.assertEqual(sp_transform(test_sample), ref_results)
