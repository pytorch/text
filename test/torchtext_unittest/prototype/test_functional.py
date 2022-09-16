import os

import torch
import torchtext.data as data
from torchtext.prototype.transforms import basic_english_normalize

from ..common.torchtext_test_case import TorchtextTestCase


class TestFunctional(TorchtextTestCase):
    def test_BasicEnglishNormalize(self) -> None:
        test_sample = "'\".<br />,()!?;:   Basic English Normalization for a Line of Text   '\".<br />,()!?;:"
        ref_results = [
            "'",
            ".",
            ",",
            "(",
            ")",
            "!",
            "?",
            "basic",
            "english",
            "normalization",
            "for",
            "a",
            "line",
            "of",
            "text",
            "'",
            ".",
            ",",
            "(",
            ")",
            "!",
            "?",
        ]

        basic_eng_norm = basic_english_normalize()
        experimental_eager_tokens = basic_eng_norm(test_sample)

        jit_basic_eng_norm = torch.jit.script(basic_eng_norm)
        experimental_jit_tokens = jit_basic_eng_norm(test_sample)

        basic_english_tokenizer = data.get_tokenizer("basic_english")
        eager_tokens = basic_english_tokenizer(test_sample)

        assert not basic_eng_norm.is_jitable
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        assert basic_eng_norm.__prepare_scriptable__().is_jitable

        self.assertEqual(experimental_jit_tokens, ref_results)
        self.assertEqual(eager_tokens, ref_results)
        self.assertEqual(experimental_eager_tokens, ref_results)

    def test_basicEnglishNormalize_load_and_save(self) -> None:
        test_sample = "'\".<br />,()!?;:   Basic English Normalization for a Line of Text   '\".<br />,()!?;:"
        ref_results = [
            "'",
            ".",
            ",",
            "(",
            ")",
            "!",
            "?",
            "basic",
            "english",
            "normalization",
            "for",
            "a",
            "line",
            "of",
            "text",
            "'",
            ".",
            ",",
            "(",
            ")",
            "!",
            "?",
        ]

        with self.subTest("pybind"):
            save_path = os.path.join(self.test_dir, "ben_pybind.pt")
            ben = basic_english_normalize()
            torch.save(ben, save_path)
            loaded_ben = torch.load(save_path)
            self.assertEqual(loaded_ben(test_sample), ref_results)

        with self.subTest("torchscript"):
            save_path = os.path.join(self.test_dir, "ben_torchscrip.pt")
            # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
            # Not expect users to use the torchbind version on eager mode but still need a CI test here.
            ben = basic_english_normalize().__prepare_scriptable__()
            torch.save(ben, save_path)
            loaded_ben = torch.load(save_path)
            self.assertEqual(loaded_ben(test_sample), ref_results)
