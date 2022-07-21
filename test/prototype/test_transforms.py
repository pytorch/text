import os
import shutil
import tempfile
from unittest.mock import patch

import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.prototype.transforms import (
    sentencepiece_processor,
    sentencepiece_tokenizer,
    VectorTransform,
    MaskTransform,
    T5Transform,
)
from torchtext.prototype.vectors import FastText

from ..common.parameterized_utils import nested_params


class TestTransforms(TorchtextTestCase):
    def test_sentencepiece_processor(self):
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

    def test_sentencepiece_tokenizer(self):
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

    def test_vector_transform(self):
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

    def test_sentencepiece_load_and_save(self):
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


class TestMaskTransform(TorchtextTestCase):

    """
    Testing under these assumed conditions:

        Vocab maps the following tokens to the following ids:
            ['a', 'b', 'c', 'd', '[PAD]', '[MASK]', '[BOS]'] -> [0, 1, 2, 3, 4, 5, 6]

        The sample token sequences are:
            [["[BOS]", "a", "b", "c", "d"],
            ["[BOS]", "a", "b", "[PAD]", "[PAD]"]]
    """

    sample_token_ids = torch.tensor([[6, 0, 1, 2, 3], [6, 0, 1, 4, 4]])

    vocab_len = 7
    pad_idx = 4
    mask_idx = 5
    bos_idx = 6

    @nested_params([0.0, 1.0])
    def test_mask_transform_probs(self, test_mask_prob):

        # We pass (vocab_len - 1) into MaskTransform to test masking with a random token.
        # This modifies the distribution from which token ids are randomly selected such that the
        # largest token id availible for selection is 1 less than the actual largest token id in our
        # vocab, which we've assigned to the [BOS] token. This allows us to test random replacement
        # by ensuring that when the first token ([BOS]) in the first sample sequence is selected for random replacement,
        # we know with certainty the token it is replaced with is different from the [BOS] token.
        # In practice, however, the actual vocab length should be provided as the input parameter so that random
        # replacement selects from all possible tokens in the vocab.
        mask_transform = MaskTransform(
            self.vocab_len - 1, self.mask_idx, self.bos_idx, self.pad_idx, mask_bos=False, mask_prob=test_mask_prob
        )

        # when mask_prob = 0, we expect the first token of the first sample sequence to be chosen for replacement
        if test_mask_prob == 0.0:

            # when mask_mask_prob, rand_mask_prob = 0,0 no tokens should change
            with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                self.assertEqual(self.sample_token_ids, masked_tokens)

            # when mask_mask_prob, rand_mask_prob = 0,1 we expect all tokens selected for replacement to be
            # changed to a random token_id
            with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 1.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)

                # first token in first sequence should be different
                self.assertNotEqual(masked_tokens[0, 0], self.sample_token_ids[0, 0])
                # replaced token id should still be in vocab, not including [BOS]
                assert masked_tokens[0, 0] in range(self.vocab_len - 1)

                # all other tokens except for first token of first sequence should remain the same
                self.assertEqual(self.sample_token_ids[0, 1:], masked_tokens[0, 1:])
                self.assertEqual(self.sample_token_ids[1], masked_tokens[1])

            # when mask_mask_prob, rand_mask_prob = 1,0 we expect all tokens selected for replacement to be changed to [MASK]
            with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 1.0), patch(
                "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                exp_tokens = torch.tensor([[5, 0, 1, 2, 3], [6, 0, 1, 4, 4]])
                self.assertEqual(exp_tokens, masked_tokens)

        # when mask_prob = 1, we expect all tokens that are not [BOS] or [PAD] to be chosen for replacement
        # (under the default condition that mask_transform.mask_bos=False)
        if test_mask_prob == 1.0:

            # when mask_mask_prob, rand_mask_prob = 0,0 no tokens should change
            with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                self.assertEqual(self.sample_token_ids, masked_tokens)

            # when mask_mask_prob, rand_mask_prob = 0,1 we expect all tokens selected for replacement
            # to be changed to random token_ids. It is possible that the randomly selected token id is the same
            # as the original token id, however we know deterministically that [BOS] and [PAD] tokens
            # in the sequences will remain unchanged.
            with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 1.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                self.assertEqual(masked_tokens[:, 0], 6 * torch.ones_like(masked_tokens[:, 0]))
                self.assertEqual(masked_tokens[1, 3:], 4 * torch.ones_like(masked_tokens[1, 3:]))

            # when mask_mask_prob, rand_mask_prob = 1,0 we expect all tokens selected for replacement to be changed to [MASK]
            with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 1.0), patch(
                "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                exp_tokens = torch.tensor([[6, 5, 5, 5, 5], [6, 5, 5, 4, 4]])
                self.assertEqual(exp_tokens, masked_tokens)

    def test_mask_transform_mask_bos(self):
        # MaskTransform has boolean parameter mask_bos to indicate whether or not [BOS] tokens
        # should be eligible for replacement. The above tests of MaskTransform are under default value
        # mask_bos = False. Here we test the case where mask_bos = True
        mask_transform = MaskTransform(
            self.vocab_len - 1, self.mask_idx, self.bos_idx, self.pad_idx, mask_bos=True, mask_prob=1.0
        )

        # when mask_mask_prob, rand_mask_prob = 1,0 we expect all tokens selected for replacement to be changed to [MASK]
        with patch("torchtext.prototype.transforms.MaskTransform.mask_mask_prob", 1.0), patch(
            "torchtext.prototype.transforms.MaskTransform.rand_mask_prob", 0.0
        ):
            masked_tokens, _, _ = mask_transform(self.sample_token_ids)
            exp_tokens = torch.tensor([[5, 5, 5, 5, 5], [5, 5, 5, 4, 4]])
            self.assertEqual(exp_tokens, masked_tokens)

    def _t5tokenizer(self, test_scripting):
        asset_name = "https://huggingface.co/t5-base/resolve/main/spiece.model"
        asset_path = get_asset_path(asset_name)
        transform = T5Transform(asset_path, max_seq_len=512, eos_idx=1, padding_idx=0)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(["Hello World!, how are you?"])
        expected = torch.tensor([[8774, 1150, 55, 6, 149, 33, 25, 58, 1]])
        self.assertEqual(actual, expected)

        actual = transform("Hello World!, how are you?")
        expected = torch.tensor([8774, 1150, 55, 6, 149, 33, 25, 58, 1])
        self.assertEqual(actual, expected)

    def test_t5tokenizer(self):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._t5tokenizer(test_scripting=False)

    def test_t5tokenizer_jit(self):
        """test tokenization with scripting on single sentence input as well as batch on sentences"""
        self._t5tokenizer(test_scripting=True)
