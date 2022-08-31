import os
import shutil
import tempfile

import torch
from torchtext.prototype.transforms import (
    sentencepiece_processor,
    sentencepiece_tokenizer,
    VectorTransform,
)
from torchtext.prototype.vectors import FastText
from torchtext_unittest.common.assets import get_asset_path
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase


class TestTransforms(TorchtextTestCase):
    def test_sentencepiece_processor(self) -> None:
        model_path = get_asset_path("spm_example.model")
        spm_transform = sentencepiece_processor(model_path)
        jit_spm_transform = torch.jit.script(spm_transform)
        test_sample = "SentencePiece is an unsupervised text tokenizer and detokenizer"
        ref_results = [
            15340,
            4286,
            981,
            1207,
            1681,
            17,
            84,
            684,
            8896,
            5366,
            144,
            3689,
            9,
            5602,
            12114,
            6,
            560,
            649,
            5602,
            12114,
        ]
        self.assertEqual(spm_transform(test_sample), ref_results)
        self.assertEqual(jit_spm_transform(test_sample), ref_results)
        self.assertEqual(spm_transform.decode(ref_results), test_sample)
        self.assertEqual(jit_spm_transform.decode(ref_results), test_sample)

    def test_sentencepiece_tokenizer(self) -> None:
        model_path = get_asset_path("spm_example.model")
        spm_tokenizer = sentencepiece_tokenizer(model_path)
        jit_spm_tokenizer = torch.jit.script(spm_tokenizer)
        test_sample = "SentencePiece is an unsupervised text tokenizer and detokenizer"
        ref_results = [
            "\u2581Sent",
            "ence",
            "P",
            "ie",
            "ce",
            "\u2581is",
            "\u2581an",
            "\u2581un",
            "super",
            "vis",
            "ed",
            "\u2581text",
            "\u2581to",
            "ken",
            "izer",
            "\u2581and",
            "\u2581de",
            "to",
            "ken",
            "izer",
        ]

        self.assertEqual(spm_tokenizer(test_sample), ref_results)
        self.assertEqual(spm_tokenizer.decode(ref_results), test_sample)
        self.assertEqual(jit_spm_tokenizer(test_sample), ref_results)
        self.assertEqual(jit_spm_tokenizer.decode(ref_results), test_sample)

    def test_vector_transform(self) -> None:
        asset_name = "wiki.en.vec"
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)
            vector_transform = VectorTransform(FastText(root=dir_name, validate_file=False))
            jit_vector_transform = torch.jit.script(vector_transform)
            # The first 3 entries in each vector.
            expected_fasttext_simple_en = torch.tensor(
                [[-0.065334, -0.093031, -0.017571], [-0.32423, -0.098845, -0.0073467]]
            )
            self.assertEqual(vector_transform(["the", "world"])[:, 0:3], expected_fasttext_simple_en)
            self.assertEqual(jit_vector_transform(["the", "world"])[:, 0:3], expected_fasttext_simple_en)

    def test_sentencepiece_load_and_save(self) -> None:
        model_path = get_asset_path("spm_example.model")
        input = "SentencePiece is an unsupervised text tokenizer and detokenizer"
        expected = [
            "▁Sent",
            "ence",
            "P",
            "ie",
            "ce",
            "▁is",
            "▁an",
            "▁un",
            "super",
            "vis",
            "ed",
            "▁text",
            "▁to",
            "ken",
            "izer",
            "▁and",
            "▁de",
            "to",
            "ken",
            "izer",
        ]

        with self.subTest("pybind"):
            save_path = os.path.join(self.test_dir, "spm_pybind.pt")
            spm = sentencepiece_tokenizer((model_path))
            torch.save(spm, save_path)
            loaded_spm = torch.load(save_path)
            self.assertEqual(expected, loaded_spm(input))

        with self.subTest("torchscript"):
            save_path = os.path.join(self.test_dir, "spm_torchscript.pt")
            # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
            # Not expect users to use the torchbind version on eager mode but still need a CI test here.
            spm = sentencepiece_tokenizer((model_path)).__prepare_scriptable__()
            torch.save(spm, save_path)
            loaded_spm = torch.load(save_path)
            self.assertEqual(expected, loaded_spm(input))
