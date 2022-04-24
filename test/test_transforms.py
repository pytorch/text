import os
from collections import OrderedDict

import torch
from torchtext import transforms
from torchtext.vocab import vocab

from .common.assets import get_asset_path
from .common.torchtext_test_case import TorchtextTestCase


class TestTransforms(TorchtextTestCase):
    def _spmtokenizer(self, test_scripting):
        asset_name = "spm_example.model"
        asset_path = get_asset_path(asset_name)
        transform = transforms.SentencePieceTokenizer(asset_path)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(["Hello World!, how are you?"])
        expected = [["▁Hello", "▁World", "!", ",", "▁how", "▁are", "▁you", "?"]]
        self.assertEqual(actual, expected)

        actual = transform("Hello World!, how are you?")
        expected = ["▁Hello", "▁World", "!", ",", "▁how", "▁are", "▁you", "?"]
        self.assertEqual(actual, expected)

    def test_spmtokenizer(self):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._spmtokenizer(test_scripting=False)

    def test_spmtokenizer_jit(self):
        """test tokenization with scripting on single sentence input as well as batch on sentences"""
        self._spmtokenizer(test_scripting=True)

    def _vocab_transform(self, test_scripting):
        vocab_obj = vocab(OrderedDict([("a", 1), ("b", 1), ("c", 1)]))
        transform = transforms.VocabTransform(vocab_obj)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform([["a", "b", "c"]])
        expected = [[0, 1, 2]]
        self.assertEqual(actual, expected)

        actual = transform(["a", "b", "c"])
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

    def test_vocab_transform(self):
        """test token to indices on both sequence of input tokens as well as batch of sequence"""
        self._vocab_transform(test_scripting=False)

    def test_vocab_transform_jit(self):
        """test token to indices with scripting on both sequence of input tokens as well as batch of sequence"""
        self._vocab_transform(test_scripting=True)

    def _totensor(self, test_scripting):
        padding_value = 0
        transform = transforms.ToTensor(padding_value=padding_value)
        if test_scripting:
            transform = torch.jit.script(transform)
        input = [[1, 2], [1, 2, 3]]

        actual = transform(input)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

        input = [1, 2]
        actual = transform(input)
        expected = torch.tensor([1, 2], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_totensor(self):
        """test tensorization on both single sequence and batch of sequence"""
        self._totensor(test_scripting=False)

    def test_totensor_jit(self):
        """test tensorization with scripting on both single sequence and batch of sequence"""
        self._totensor(test_scripting=True)

    def _labeltoindex(self, test_scripting):
        label_names = ["test", "label", "indices"]
        transform = transforms.LabelToIndex(label_names=label_names)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

        with self.assertRaises(RuntimeError):
            transform(["OOV"])

        transform = transforms.LabelToIndex(label_names=label_names, sort_names=True)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [2, 1, 0]
        self.assertEqual(actual, expected)

        actual = transform("indices")
        expected = 0
        self.assertEqual(actual, expected)

        asset_name = "label_names.txt"
        asset_path = get_asset_path(asset_name)
        transform = transforms.LabelToIndex(label_path=asset_path)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

    def test_labeltoindex(self):
        """test labe to ids on single label input as well as batch of labels"""
        self._labeltoindex(test_scripting=False)

    def test_labeltoindex_jit(self):
        """test labe to ids with scripting on single label input as well as batch of labels"""
        self._labeltoindex(test_scripting=True)

    def _truncate(self, test_scripting):
        max_seq_len = 2
        transform = transforms.Truncate(max_seq_len=max_seq_len)
        if test_scripting:
            transform = torch.jit.script(transform)

        input = [[1, 2], [1, 2, 3]]
        actual = transform(input)
        expected = [[1, 2], [1, 2]]
        self.assertEqual(actual, expected)

        input = [1, 2, 3]
        actual = transform(input)
        expected = [1, 2]
        self.assertEqual(actual, expected)

        input = [["a", "b"], ["a", "b", "c"]]
        actual = transform(input)
        expected = [["a", "b"], ["a", "b"]]
        self.assertEqual(actual, expected)

        input = ["a", "b", "c"]
        actual = transform(input)
        expected = ["a", "b"]
        self.assertEqual(actual, expected)

    def test_truncate(self):
        """test truncation on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=False)

    def test_truncate_jit(self):
        """test truncation with scripting on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=True)

    def _add_token(self, test_scripting):
        token_id = 0
        transform = transforms.AddToken(token_id, begin=True)
        if test_scripting:
            transform = torch.jit.script(transform)
        input = [[1, 2], [1, 2, 3]]

        actual = transform(input)
        expected = [[0, 1, 2], [0, 1, 2, 3]]
        self.assertEqual(actual, expected)

        transform = transforms.AddToken(token_id, begin=False)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(input)
        expected = [[1, 2, 0], [1, 2, 3, 0]]
        self.assertEqual(actual, expected)

        input = [1, 2]
        actual = transform(input)
        expected = [1, 2, 0]
        self.assertEqual(actual, expected)

        token_id = "0"
        transform = transforms.AddToken(token_id, begin=True)
        if test_scripting:
            transform = torch.jit.script(transform)
        input = [["1", "2"], ["1", "2", "3"]]

        actual = transform(input)
        expected = [["0", "1", "2"], ["0", "1", "2", "3"]]
        self.assertEqual(actual, expected)

        transform = transforms.AddToken(token_id, begin=False)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(input)
        expected = [["1", "2", "0"], ["1", "2", "3", "0"]]
        self.assertEqual(actual, expected)

        input = ["1", "2"]
        actual = transform(input)
        expected = ["1", "2", "0"]
        self.assertEqual(actual, expected)

    def test_add_token(self):
        self._add_token(test_scripting=False)

    def test_add_token_jit(self):
        self._add_token(test_scripting=True)

    def _pad_transform(self, test_scripting):
        """
        Test padding transform on 1D and 2D tensors.
        When max_length < tensor length at dim -1, this should be a no-op.
        Otherwise the tensor should be padded to max_length in dim -1.
        """

        input_1d_tensor = torch.ones(5)
        input_2d_tensor = torch.ones((8, 5))
        pad_long = transforms.PadTransform(max_length=7, pad_value=0)
        if test_scripting:
            pad_long = torch.jit.script(pad_long)
        padded_1d_tensor_actual = pad_long(input_1d_tensor)
        padded_1d_tensor_expected = torch.cat([torch.ones(5), torch.zeros(2)])
        torch.testing.assert_close(
            padded_1d_tensor_actual,
            padded_1d_tensor_expected,
            msg=f"actual: {padded_1d_tensor_actual}, expected: {padded_1d_tensor_expected}",
        )

        padded_2d_tensor_actual = pad_long(input_2d_tensor)
        padded_2d_tensor_expected = torch.cat([torch.ones(8, 5), torch.zeros(8, 2)], axis=-1)
        torch.testing.assert_close(
            padded_2d_tensor_actual,
            padded_2d_tensor_expected,
            msg=f"actual: {padded_2d_tensor_actual}, expected: {padded_2d_tensor_expected}",
        )

        pad_short = transforms.PadTransform(max_length=3, pad_value=0)
        if test_scripting:
            pad_short = torch.jit.script(pad_short)
        padded_1d_tensor_actual = pad_short(input_1d_tensor)
        padded_1d_tensor_expected = input_1d_tensor
        torch.testing.assert_close(
            padded_1d_tensor_actual,
            padded_1d_tensor_expected,
            msg=f"actual: {padded_1d_tensor_actual}, expected: {padded_1d_tensor_expected}",
        )

        padded_2d_tensor_actual = pad_short(input_2d_tensor)
        padded_2d_tensor_expected = input_2d_tensor
        torch.testing.assert_close(
            padded_2d_tensor_actual,
            padded_2d_tensor_expected,
            msg=f"actual: {padded_2d_tensor_actual}, expected: {padded_2d_tensor_expected}",
        )

    def test_pad_transform(self):
        self._pad_transform(test_scripting=False)

    def test_pad_transform_jit(self):
        self._pad_transform(test_scripting=True)

    def _str_to_int_transform(self, test_scripting):
        """
        Test StrToIntTransform on list and list of lists.
        The result should be the same shape as the input but with all strings converted to ints.
        """
        input_1d_string_list = ["1", "2", "3", "4", "5"]
        input_2d_string_list = [["1", "2", "3"], ["4", "5", "6"]]

        str_to_int = transforms.StrToIntTransform()
        if test_scripting:
            str_to_int = torch.jit.script(str_to_int)

        expected_1d_int_list = [1, 2, 3, 4, 5]
        actual_1d_int_list = str_to_int(input_1d_string_list)
        self.assertListEqual(expected_1d_int_list, actual_1d_int_list)

        expected_2d_int_list = [[1, 2, 3], [4, 5, 6]]
        actual_2d_int_list = str_to_int(input_2d_string_list)
        for i in range(len(expected_2d_int_list)):
            self.assertListEqual(expected_2d_int_list[i], actual_2d_int_list[i])

    def test_str_to_int_transform(self):
        self._str_to_int_transform(test_scripting=False)

    def test_str_to_int_transform_jit(self):
        self._str_to_int_transform(test_scripting=True)


class TestSequential(TorchtextTestCase):
    def _sequential(self, test_scripting):
        max_seq_len = 3
        padding_val = 0
        transform = transforms.Sequential(
            transforms.Truncate(max_seq_len=max_seq_len),
            transforms.ToTensor(padding_value=padding_val, dtype=torch.long),
        )

        if test_scripting:
            transform = torch.jit.script(transform)

        input = [[1, 2, 3], [1, 2, 3]]

        actual = transform(input)
        expected = torch.tensor(input)
        torch.testing.assert_close(actual, expected)

    def test_sequential(self):
        """test pipelining transforms using Sequential transform"""
        self._sequential(test_scripting=False)

    def test_sequential_jit(self):
        """test pipelining transforms using Sequential transform, ensuring the composite transform is scriptable"""
        self._sequential(test_scripting=True)


class TestGPT2BPETokenizer(TorchtextTestCase):
    def _load_tokenizer(self, test_scripting):
        encoder_json = "gpt2_bpe_encoder.json"
        bpe_vocab = "gpt2_bpe_vocab.bpe"
        tokenizer = transforms.GPT2BPETokenizer(
            encoder_json_path=get_asset_path(encoder_json),
            vocab_bpe_path=get_asset_path(bpe_vocab),
        )
        if test_scripting:
            tokenizer = torch.jit.script(tokenizer)
        return tokenizer

    def _gpt2_bpe_tokenizer(self, tokenizer):
        sample_texts = [
            "Hello World!, how are you?",
            "Hélló  WoŕlḊ¿",
            "Respublica superiorem",
            "Avdija Vršajević în",
        ]

        expected_token_ids = [
            ["15496", "2159", "28265", "703", "389", "345", "30"],
            ["39", "2634", "297", "10205", "220", "22173", "129", "243", "75", "41585", "232", "126", "123"],
            ["4965", "11377", "64", "2208", "72", "29625"],
            ["7355", "67", "34655", "569", "81", "32790", "1228", "1990", "72", "38325", "6184", "106", "77"],
        ]

        # test batch of sentences
        self.assertEqual(tokenizer(sample_texts), expected_token_ids)

        # test individual sentences
        for idx, txt in enumerate(sample_texts):
            self.assertEqual(tokenizer(txt), expected_token_ids[idx])

    def test_gpt2_bpe_tokenizer(self):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._gpt2_bpe_tokenizer(self._load_tokenizer(test_scripting=False))

    def test_gpt2_bpe_tokenizer_jit(self):
        """test tokenization with scripting on single sentence input as well as batch on sentences"""
        self._gpt2_bpe_tokenizer(self._load_tokenizer(test_scripting=True))

    def test_gpt2_bpe_tokenizer_save_load_pybind(self):
        tokenizer = self._load_tokenizer(test_scripting=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_pybind.pt")
        torch.save(tokenizer, tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._gpt2_bpe_tokenizer((loaded_tokenizer))

    def test_gpt2_bpe_tokenizer_save_load_torchscript(self):
        tokenizer = self._load_tokenizer(test_scripting=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_torchscript.pt")
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        torch.save(tokenizer.__prepare_scriptable__(), tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._gpt2_bpe_tokenizer((loaded_tokenizer))


class TestCLIPTokenizer(TorchtextTestCase):
    def _load_tokenizer(self, init_using_merge_only: bool, test_scripting: bool):
        encoder_json = "clip_encoder.json"
        bpe_vocab = "clip_vocab.bpe"
        num_merges = (
            49152 - 256 - 2
        )  # https://github.com/mlfoundations/open_clip/blob/57b3e8ea6ad6bfc2974203945f8fd577e0659468/src/clip/tokenizer.py#L67
        if init_using_merge_only:
            tokenizer = transforms.CLIPTokenizer(
                merges_path=get_asset_path(bpe_vocab),
                num_merges=num_merges,
            )
        else:
            tokenizer = transforms.CLIPTokenizer(
                encoder_json_path=get_asset_path(encoder_json),
                merges_path=get_asset_path(bpe_vocab),
            )
        if test_scripting:
            tokenizer = torch.jit.script(tokenizer)
        return tokenizer

    def _clip_tokenizer(self, tokenizer):
        sample_texts = [
            "Hello World!, how are you?",
            "<|startoftext|> the quick brown fox jumped over the lazy dog <|endoftext|>",
            "Awaiting their due award... Photo by Frederick (FN) Noronha. Copyleft. Creative Commons 3.0. Non-commercial. Attribution. May be copied for non-commercial purposes. For other purposes, contact fn at goa-india.org",
        ]

        expected_token_ids = [
            ["3306", "1002", "29325", "829", "631", "592", "286"],
            ["49406", "518", "3712", "2866", "3240", "16901", "962", "518", "10753", "1929", "49407"],
            [
                "14872",
                "911",
                "2887",
                "2047",
                "678",
                "1125",
                "638",
                "18570",
                "263",
                "21763",
                "264",
                "1062",
                "521",
                "1429",
                "269",
                "11376",
                "1823",
                "269",
                "4450",
                "16653",
                "274",
                "269",
                "271",
                "269",
                "3353",
                "268",
                "6287",
                "269",
                "24624",
                "740",
                "269",
                "1270",
                "655",
                "36770",
                "556",
                "3353",
                "268",
                "6287",
                "22020",
                "269",
                "556",
                "1010",
                "22020",
                "267",
                "3523",
                "21763",
                "536",
                "14399",
                "268",
                "1762",
                "269",
                "5593",
            ],
        ]

        # test batch of sentences
        self.assertEqual(tokenizer(sample_texts), expected_token_ids)

        # test individual sentences
        for idx, txt in enumerate(sample_texts):
            self.assertEqual(tokenizer(txt), expected_token_ids[idx])

    def test_clip_tokenizer(self):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._clip_tokenizer(self._load_tokenizer(init_using_merge_only=True, test_scripting=False))
        self._clip_tokenizer(self._load_tokenizer(init_using_merge_only=False, test_scripting=False))

    def test_clip_tokenizer_jit(self):
        """test tokenization with scripting on single sentence input as well as batch on sentences"""
        self._clip_tokenizer(self._load_tokenizer(init_using_merge_only=True, test_scripting=True))
        self._clip_tokenizer(self._load_tokenizer(init_using_merge_only=False, test_scripting=True))

    def test_clip_tokenizer_save_load_pybind(self):
        tokenizer = self._load_tokenizer(init_using_merge_only=True, test_scripting=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_pybind.pt")
        torch.save(tokenizer, tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._clip_tokenizer((loaded_tokenizer))

    def test_clip_tokenizer_save_load_torchscript(self):
        tokenizer = self._load_tokenizer(init_using_merge_only=True, test_scripting=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_torchscript.pt")
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        torch.save(tokenizer.__prepare_scriptable__(), tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._clip_tokenizer((loaded_tokenizer))
