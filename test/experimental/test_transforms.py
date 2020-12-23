import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.transforms import (
    VectorTransform,
    sentencepiece_processor,
    sentencepiece_tokenizer,
)
from torchtext.experimental.vectors import FastText
import shutil
import tempfile
import os
import unittest
import platform


class TestTransforms(TorchtextTestCase):
    def test_sentencepiece_processor(self):
        model_path = get_asset_path('spm_example.model')
        spm_transform = sentencepiece_processor(model_path)
        jit_spm_transform = torch.jit.script(spm_transform.to_ivalue())
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        ref_results = [15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                       144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]
        self.assertEqual(spm_transform(test_sample), ref_results)
        self.assertEqual(jit_spm_transform(test_sample), ref_results)
        self.assertEqual(spm_transform.decode(ref_results), test_sample)
        self.assertEqual(jit_spm_transform.decode(ref_results), test_sample)

    def test_sentencepiece_tokenizer(self):
        model_path = get_asset_path('spm_example.model')
        spm_tokenizer = sentencepiece_tokenizer(model_path)
        jit_spm_tokenizer = torch.jit.script(spm_tokenizer.to_ivalue())
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                       '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                       '\u2581to', 'ken', 'izer', '\u2581and',
                       '\u2581de', 'to', 'ken', 'izer']

        self.assertEqual(spm_tokenizer(test_sample), ref_results)
        self.assertEqual(spm_tokenizer.decode(ref_results), test_sample)
        self.assertEqual(jit_spm_tokenizer(test_sample), ref_results)
        self.assertEqual(jit_spm_tokenizer.decode(ref_results), test_sample)

    # we separate out these errors because Windows runs into seg faults when propagating
    # exceptions from C++ using pybind11
    @unittest.skipIf(platform.system() == "Windows", "Test is known to fail on Windows.")
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
            self.assertEqual(vector_transform(['the', 'world'])[:, 0:3], expected_fasttext_simple_en)
            self.assertEqual(jit_vector_transform(['the', 'world'])[:, 0:3], expected_fasttext_simple_en)

    def test_sentencepiece_load_and_save(self):
        model_path = get_asset_path('spm_example.model')
        input = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        expected = [
            '▁Sent', 'ence', 'P', 'ie', 'ce', '▁is',
            '▁an', '▁un', 'super', 'vis', 'ed', '▁text',
            '▁to', 'ken', 'izer', '▁and',
            '▁de', 'to', 'ken', 'izer',
        ]

        with self.subTest('pybind'):
            save_path = os.path.join(self.test_dir, 'spm_pybind.pt')
            spm = sentencepiece_tokenizer((model_path))
            torch.save(spm, save_path)
            loaded_spm = torch.load(save_path)
            self.assertEqual(expected, loaded_spm(input))

        with self.subTest('torchscript'):
            save_path = os.path.join(self.test_dir, 'spm_torchscript.pt')
            spm = sentencepiece_tokenizer((model_path)).to_ivalue()
            torch.save(spm, save_path)
            loaded_spm = torch.load(save_path)
            self.assertEqual(expected, loaded_spm(input))
