import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.transforms import (
    VectorTransform,
    VocabTransform,
    SentencePieceTransform,
    SentencePieceTokenizer,
    TextSequentialTransforms,
    pretrained_spm,
)
from torchtext.experimental.vocab import vocab_from_file
from torchtext.experimental.vectors import FastText
from torchtext.data.functional import load_sp_model
import shutil
import tempfile
import os


class TestTransforms(TorchtextTestCase):
    def test_sentencepiece_transform(self):
        model_path = get_asset_path('spm_example.model')
        pipeline = TextSequentialTransforms(SentencePieceTransform(load_sp_model(model_path)))
        jit_pipeline = torch.jit.script(pipeline)
        test_sample = ['SentencePiece is an unsupervised text tokenizer and detokenizer']
        ref_results = [[15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                        144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]]
        self.assertEqual(pipeline(test_sample), ref_results)
        self.assertEqual(jit_pipeline(test_sample), ref_results)

        spm_transform = SentencePieceTransform(load_sp_model(model_path))
        self.assertEqual(spm_transform.decode([[15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366, 144, 3689,
                                               9, 5602, 12114, 6, 560, 649, 5602, 12114]]), test_sample)

    def test_sentencepiece_tokenizer(self):
        model_path = get_asset_path('spm_example.model')
        spm_tokenizer = SentencePieceTokenizer(load_sp_model(model_path))
        jit_spm_tokenizer = torch.jit.script(spm_tokenizer)
        test_sample = ['SentencePiece is an unsupervised text tokenizer and detokenizer']
        ref_results = [['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                        '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                        '\u2581to', 'ken', 'izer', '\u2581and',
                        '\u2581de', 'to', 'ken', 'izer']]

        self.assertEqual(spm_tokenizer(test_sample), ref_results)
        self.assertEqual(spm_tokenizer.decode(ref_results), test_sample)
        self.assertEqual(jit_spm_tokenizer(test_sample), ref_results)
        self.assertEqual(jit_spm_tokenizer.decode(ref_results), test_sample)

    def test_builtin_pretrained_sentencepiece_transform(self):
        spm_tokenizer = SentencePieceTokenizer(pretrained_spm())
        _path = os.path.join(self.project_root, '.data', 'text_unigram_25000.model')
        os.remove(_path)
        test_sample = ['the pretrained spm model names']
        ref_results = [['\u2581the', '\u2581pre', 'trained', '\u2581sp', 'm', '\u2581model', '\u2581names']]
        self.assertEqual(spm_tokenizer(test_sample), ref_results)

        spm_transform = SentencePieceTransform(pretrained_spm('text_bpe_25000'))
        _path = os.path.join(self.project_root, '.data', 'text_bpe_25000.model')
        os.remove(_path)
        test_sample = ['the pretrained spm model names']
        ref_results = [[13, 1465, 12824, 304, 24935, 5771, 3776]]
        self.assertEqual(spm_transform(test_sample), ref_results)

    def test_vocab_transform(self):
        asset_name = 'vocab_test2.txt'
        asset_path = get_asset_path(asset_name)
        with open(asset_path, 'r') as f:
            vocab_transform = VocabTransform(vocab_from_file(f))
            self.assertEqual(vocab_transform([['of', 'that', 'new'], ['of', 'that', 'new', 'that']]),
                             [[21, 26, 20], [21, 26, 20, 26]])
            jit_vocab_transform = torch.jit.script(vocab_transform.to_ivalue())
            self.assertEqual(jit_vocab_transform([['of', 'that', 'new'], ['of', 'that', 'new', 'that']]),
                             [[21, 26, 20], [21, 26, 20, 26]])

    def test_vector_transform(self):
        asset_name = 'wiki.en.vec'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)
            vector_transform = VectorTransform(FastText(root=dir_name, validate_file=False))
            jit_vector_transform = torch.jit.script(vector_transform.to_ivalue())
            # The first 3 entries in each vector.
            expected_fasttext_simple_en = torch.tensor([[-0.065334, -0.093031, -0.017571],
                                                        [-0.32423, -0.098845, -0.0073467]])
            self.assertEqual(vector_transform([['the', 'world']])[0][:, 0:3], expected_fasttext_simple_en)
            self.assertEqual(jit_vector_transform([['the', 'world']])[0][:, 0:3], expected_fasttext_simple_en)
