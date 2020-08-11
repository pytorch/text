import torch
from torchtext.experimental.transforms import (
    PretrainedSPTransform,
    ToLongTensor,
    TextSequentialTransforms,
)
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase


class TestTransforms(TorchtextTestCase):
    def test_sentencepiece_pretrained_pipeline(self):
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = get_asset_path('spm_example.model')
        pipeline = TextSequentialTransforms(PretrainedSPTransform(model_path), ToLongTensor)
        jit_pipeline = torch.jit.script(pipeline)
        ref_results = torch.tensor([15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                                    144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]).to(torch.long)
        self.assertEqual(pipeline(test_sample), ref_results)
        self.assertEqual(jit_pipeline(test_sample), ref_results)

        spm_transform = PretrainedSPTransform(model_path)
        self.assertEqual(spm_transform.decode([15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366, 144, 3689,
                                               9, 5602, 12114, 6, 560, 649, 5602, 12114]), test_sample)

    def test_sentencepiece_pretrained_tokenizer_transform(self):
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = get_asset_path('spm_example.model')
        spm_tokenizer = PretrainedSPTokenizer(model_path)
        jit_spm_tokenizer = torch.jit.script(spm_tokenizer)

        ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                       '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                       '\u2581to', 'ken', 'izer', '\u2581and',
                       '\u2581de', 'to', 'ken', 'izer']

        self.assertEqual(spm_tokenizer(test_sample), ref_results)
        self.assertEqual(spm_tokenizer.decode(ref_results), test_sample)
        self.assertEqual(jit_spm_tokenizer(test_sample), ref_results)
        self.assertEqual(jit_spm_tokenizer.decode(ref_results), test_sample)

    def test_builtin_pretrained_sentencepiece_transform(self):
        spm_tokenizer = PretrainedSPTokenizer()
        test_sample = 'the pretrained spm model names'
        ref_results = ['\u2581the', '\u2581pre', 'trained', '\u2581sp', 'm', '\u2581model', '\u2581names']
        self.assertEqual(spm_tokenizer(test_sample), ref_results)

        spm_transform = PretrainedSPTransform()
        test_sample = 'the pretrained spm model names'
        ref_results = [9, 1546, 18811, 2849, 61, 2759, 2202]
        self.assertEqual(spm_transform(test_sample), ref_results)
