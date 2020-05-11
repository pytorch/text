from torchtext.experimental.transforms import SentencePieceTokenizer
from ..common.torchtext_test_case import TorchtextTestCase


class TestTransforms(TorchtextTestCase):
    def test_SentencePieceTokenizer(self):

        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_tokenizer = SentencePieceTokenizer(model_path)
        ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                       '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                       '\u2581to', 'ken', 'izer', '\u2581and',
                       '\u2581de', 'to', 'ken', 'izer']
        self.assertEqual(sp_tokenizer(test_sample), ref_results)
