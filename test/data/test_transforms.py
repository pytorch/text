from ..common.torchtext_test_case import TorchtextTestCase
from torchtext.data.transforms import SentencePieceTransform


class TestUtils(TorchtextTestCase):
    def test_sentencepiece_transform(self):
        # sentencepiece transform.
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_transform = SentencePieceTransform(model_path)
        ref_results = [15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                       144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]
        self.assertEqual(sp_transform(test_sample), ref_results)
