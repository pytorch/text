import os
from collections import OrderedDict
from unittest.mock import patch

import torch
from torchtext import transforms
from torchtext.transforms import MaskTransform, RegexTokenizer
from torchtext.vocab import vocab

from .common.assets import get_asset_path
from .common.parameterized_utils import nested_params
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

    def test_spmtokenizer(self) -> None:
        """test tokenization on single sentence input as well as batch on sentences"""
        self._spmtokenizer(test_scripting=False)

    def test_spmtokenizer_jit(self) -> None:
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

    def test_vocab_transform(self) -> None:
        """test token to indices on both sequence of input tokens as well as batch of sequence"""
        self._vocab_transform(test_scripting=False)

    def test_vocab_transform_jit(self) -> None:
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

    def test_totensor(self) -> None:
        """test tensorization on both single sequence and batch of sequence"""
        self._totensor(test_scripting=False)

    def test_totensor_jit(self) -> None:
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

    def test_labeltoindex(self) -> None:
        """test labe to ids on single label input as well as batch of labels"""
        self._labeltoindex(test_scripting=False)

    def test_labeltoindex_jit(self) -> None:
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

    def test_truncate(self) -> None:
        """test truncation on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=False)

    def test_truncate_jit(self) -> None:
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

    def test_add_token(self) -> None:
        self._add_token(test_scripting=False)

    def test_add_token_jit(self) -> None:
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

    def test_pad_transform(self) -> None:
        self._pad_transform(test_scripting=False)

    def test_pad_transform_jit(self) -> None:
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

    def test_str_to_int_transform(self) -> None:
        self._str_to_int_transform(test_scripting=False)

    def test_str_to_int_transform_jit(self) -> None:
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

    def test_sequential(self) -> None:
        """test pipelining transforms using Sequential transform"""
        self._sequential(test_scripting=False)

    def test_sequential_jit(self) -> None:
        """test pipelining transforms using Sequential transform, ensuring the composite transform is scriptable"""
        self._sequential(test_scripting=True)


class TestGPT2BPETokenizer(TorchtextTestCase):
    def _load_tokenizer(self, test_scripting: bool, return_tokens: bool):
        encoder_json = "gpt2_bpe_encoder.json"
        bpe_vocab = "gpt2_bpe_vocab.bpe"
        tokenizer = transforms.GPT2BPETokenizer(
            encoder_json_path=get_asset_path(encoder_json),
            vocab_bpe_path=get_asset_path(bpe_vocab),
            return_tokens=return_tokens,
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

        expected_tokens = [
            ["Hello", "ĠWorld", "!,", "Ġhow", "Ġare", "Ġyou", "?"],
            ["H", "Ã©", "ll", "Ã³", "Ġ", "ĠWo", "Å", "ķ", "l", "á¸", "Ĭ", "Â", "¿"],
            ["Res", "public", "a", "Ġsuper", "i", "orem"],
            ["Av", "d", "ija", "ĠV", "r", "Å¡", "aj", "ev", "i", "Äĩ", "ĠÃ", "®", "n"],
        ]
        expected_token_ids = [
            ["15496", "2159", "28265", "703", "389", "345", "30"],
            ["39", "2634", "297", "10205", "220", "22173", "129", "243", "75", "41585", "232", "126", "123"],
            ["4965", "11377", "64", "2208", "72", "29625"],
            ["7355", "67", "34655", "569", "81", "32790", "1228", "1990", "72", "38325", "6184", "106", "77"],
        ]

        # test batch of sentences
        if tokenizer._return_tokens:
            self.assertEqual(tokenizer(sample_texts), expected_tokens)
        else:
            self.assertEqual(tokenizer(sample_texts), expected_token_ids)

        # test individual sentences
        for idx, txt in enumerate(sample_texts):
            if tokenizer._return_tokens:
                self.assertEqual(tokenizer(txt), expected_tokens[idx])
            else:
                self.assertEqual(tokenizer(txt), expected_token_ids[idx])

    @nested_params([True, False], [True, False])
    def test_gpt2_bpe_tokenizer(self, test_scripting, return_tokens):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._gpt2_bpe_tokenizer(self._load_tokenizer(test_scripting=test_scripting, return_tokens=return_tokens))

    def test_gpt2_bpe_tokenizer_save_load_pybind(self) -> None:
        tokenizer = self._load_tokenizer(test_scripting=False, return_tokens=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_pybind.pt")
        torch.save(tokenizer, tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._gpt2_bpe_tokenizer((loaded_tokenizer))

    def test_gpt2_bpe_tokenizer_save_load_torchscript(self) -> None:
        tokenizer = self._load_tokenizer(test_scripting=False, return_tokens=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_torchscript.pt")
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        torch.save(tokenizer.__prepare_scriptable__(), tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._gpt2_bpe_tokenizer((loaded_tokenizer))


class TestCLIPTokenizer(TorchtextTestCase):
    def _load_tokenizer(self, init_using_merge_only: bool, test_scripting: bool, return_tokens: bool):
        encoder_json = "clip_encoder.json"
        bpe_vocab = "clip_vocab.bpe"
        num_merges = (
            49152 - 256 - 2
        )  # https://github.com/mlfoundations/open_clip/blob/57b3e8ea6ad6bfc2974203945f8fd577e0659468/src/clip/tokenizer.py#L67
        if init_using_merge_only:
            tokenizer = transforms.CLIPTokenizer(
                merges_path=get_asset_path(bpe_vocab),
                num_merges=num_merges,
                return_tokens=return_tokens,
            )
        else:
            tokenizer = transforms.CLIPTokenizer(
                encoder_json_path=get_asset_path(encoder_json),
                merges_path=get_asset_path(bpe_vocab),
                return_tokens=return_tokens,
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

        expected_tokens = [
            ["hello</w>", "world</w>", "!,</w>", "how</w>", "are</w>", "you</w>", "?</w>"],
            [
                "<|startoftext|>",
                "the</w>",
                "quick</w>",
                "brown</w>",
                "fox</w>",
                "jumped</w>",
                "over</w>",
                "the</w>",
                "lazy</w>",
                "dog</w>",
                "<|endoftext|>",
            ],
            [
                "awaiting</w>",
                "their</w>",
                "due</w>",
                "award</w>",
                "...</w>",
                "photo</w>",
                "by</w>",
                "frederick</w>",
                "(</w>",
                "fn</w>",
                ")</w>",
                "nor",
                "on",
                "ha</w>",
                ".</w>",
                "copy",
                "left</w>",
                ".</w>",
                "creative</w>",
                "commons</w>",
                "3</w>",
                ".</w>",
                "0</w>",
                ".</w>",
                "non</w>",
                "-</w>",
                "commercial</w>",
                ".</w>",
                "attribu",
                "tion</w>",
                ".</w>",
                "may</w>",
                "be</w>",
                "copied</w>",
                "for</w>",
                "non</w>",
                "-</w>",
                "commercial</w>",
                "purposes</w>",
                ".</w>",
                "for</w>",
                "other</w>",
                "purposes</w>",
                ",</w>",
                "contact</w>",
                "fn</w>",
                "at</w>",
                "goa</w>",
                "-</w>",
                "india</w>",
                ".</w>",
                "org</w>",
            ],
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

        if tokenizer._return_tokens:
            self.assertEqual(tokenizer(sample_texts), expected_tokens)
        else:
            self.assertEqual(tokenizer(sample_texts), expected_token_ids)

        # test individual sentences
        for idx, txt in enumerate(sample_texts):
            if tokenizer._return_tokens:
                self.assertEqual(tokenizer(txt), expected_tokens[idx])
            else:
                self.assertEqual(tokenizer(txt), expected_token_ids[idx])

    @nested_params([True, False], [True, False], [True, False])
    def test_clip_tokenizer(self, init_using_merge_only, test_scripting, return_tokens):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._clip_tokenizer(
            self._load_tokenizer(
                init_using_merge_only=init_using_merge_only, test_scripting=test_scripting, return_tokens=return_tokens
            )
        )

    def test_clip_tokenizer_save_load_pybind(self) -> None:
        tokenizer = self._load_tokenizer(init_using_merge_only=True, test_scripting=False, return_tokens=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_pybind.pt")
        torch.save(tokenizer, tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._clip_tokenizer((loaded_tokenizer))

    def test_clip_tokenizer_save_load_torchscript(self) -> None:
        tokenizer = self._load_tokenizer(init_using_merge_only=True, test_scripting=False, return_tokens=False)
        tokenizer_path = os.path.join(self.test_dir, "gpt2_tokenizer_torchscript.pt")
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        torch.save(tokenizer.__prepare_scriptable__(), tokenizer_path)
        loaded_tokenizer = torch.load(tokenizer_path)
        self._clip_tokenizer((loaded_tokenizer))


class TestBERTTokenizer(TorchtextTestCase):
    def _load_tokenizer(self, test_scripting: bool, do_lower_case: bool, return_tokens: bool):
        if do_lower_case:
            vocab_file = "bert_base_uncased_vocab.txt"
        else:
            vocab_file = "bert_base_cased_vocab.txt"

        tokenizer = transforms.BERTTokenizer(
            vocab_path=get_asset_path(vocab_file),
            do_lower_case=do_lower_case,
            return_tokens=return_tokens,
        )
        if test_scripting:
            tokenizer = torch.jit.script(tokenizer)
        return tokenizer

    def _bert_tokenizer(self, tokenizer, do_lower_case):
        sample_texts = [
            "Hello World!, how are you?",
            "Hélló  WoŕlḊ¿",
            "Respublica superiorem",
            "Avdija Vršajević în",
        ]

        if do_lower_case:
            expected_tokens = [
                ["hello", "world", "!", ",", "how", "are", "you", "?"],
                ["hello", "world", "¿"],
                ["res", "##pu", "##bl", "##ica", "superior", "##em"],
                ["av", "##di", "##ja", "vr", "##sa", "##jevic", "in"],
            ]
            expected_token_ids = [
                ["7592", "2088", "999", "1010", "2129", "2024", "2017", "1029"],
                ["7592", "2088", "1094"],
                ["24501", "14289", "16558", "5555", "6020", "6633"],
                ["20704", "4305", "3900", "27830", "3736", "26782", "1999"],
            ]

        else:
            expected_tokens = [
                ["Hello", "World", "!", ",", "how", "are", "you", "?"],
                ["H", "##é", "##ll", "##ó", "[UNK]", "¿"],
                ["Re", "##sp", "##ub", "##lica", "superior", "##em"],
                ["A", "##v", "##di", "##ja", "V", "##r", "##ša", "##je", "##vić", "î", "##n"],
            ]
            expected_token_ids = [
                ["8667", "1291", "106", "117", "1293", "1132", "1128", "136"],
                ["145", "2744", "2339", "7774", "100", "225"],
                ["11336", "20080", "10354", "9538", "7298", "5521"],
                ["138", "1964", "3309", "3174", "159", "1197", "23834", "5561", "10225", "260", "1179"],
            ]

        # test batch of sentences
        if tokenizer._return_tokens:
            self.assertEqual(tokenizer(sample_texts), expected_tokens)
        else:
            self.assertEqual(tokenizer(sample_texts), expected_token_ids)

        # test individual sentences
        for idx, txt in enumerate(sample_texts):
            if tokenizer._return_tokens:
                self.assertEqual(tokenizer(txt), expected_tokens[idx])
            else:
                self.assertEqual(tokenizer(txt), expected_token_ids[idx])

    @nested_params([True, False], [True, False], [True, False])
    def test_bert_tokenizer(self, test_scripting, do_lower_case, return_tokens):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._bert_tokenizer(
            self._load_tokenizer(
                test_scripting=test_scripting, do_lower_case=do_lower_case, return_tokens=return_tokens
            ),
            do_lower_case=do_lower_case,
        )

    @nested_params([True, False], [True, False], [True, False])
    def test_bert_tokenizer_save_load(self, test_scripting, do_lower_case, return_tokens):
        """test saving and loading of BERT tokenizer both for scripted and non-scripted version"""
        tokenizer = self._load_tokenizer(
            test_scripting=test_scripting, do_lower_case=do_lower_case, return_tokens=return_tokens
        )
        tokenizer_path = os.path.join(self.test_dir, "bert_tokenizer_pybind.pt")
        if test_scripting:
            torch.jit.save(tokenizer, tokenizer_path)
            loaded_tokenizer = torch.jit.load(tokenizer_path)
            self._bert_tokenizer((loaded_tokenizer), do_lower_case=do_lower_case)
        else:
            torch.save(tokenizer, tokenizer_path)
            loaded_tokenizer = torch.load(tokenizer_path)
            self._bert_tokenizer((loaded_tokenizer), do_lower_case=do_lower_case)


class TestRegexTokenizer(TorchtextTestCase):
    test_sample = "'\".<br />,()!?;:   Basic Regex Tokenization for a Line of Text   '\".<br />,()!?;:"
    ref_results = [
        "'",
        ".",
        ",",
        "(",
        ")",
        "!",
        "?",
        "Basic",
        "Regex",
        "Tokenization",
        "for",
        "a",
        "Line",
        "of",
        "Text",
        "'",
        ".",
        ",",
        "(",
        ")",
        "!",
        "?",
    ]
    patterns_list = [
        (r"\'", " '  "),
        (r"\"", ""),
        (r"\.", " . "),
        (r"<br \/>", " "),
        (r",", " , "),
        (r"\(", " ( "),
        (r"\)", " ) "),
        (r"\!", " ! "),
        (r"\?", " ? "),
        (r"\;", " "),
        (r"\:", " "),
        (r"\s+", " "),
    ]

    def test_regex_tokenizer(self) -> None:
        r_tokenizer = RegexTokenizer(self.patterns_list)
        eager_tokens = r_tokenizer(self.test_sample)

        jit_r_tokenizer = torch.jit.script(r_tokenizer)
        jit_tokens = jit_r_tokenizer(self.test_sample)

        assert not r_tokenizer.is_jitable
        # Call the __prepare_scriptable__() func and convert the operator to the torchbind version
        # We don't expect users to use the torchbind version on eager mode but still need a CI test here.
        assert r_tokenizer.__prepare_scriptable__().is_jitable

        self.assertEqual(eager_tokens, self.ref_results)
        self.assertEqual(jit_tokens, self.ref_results)

    def test_regex_tokenizer_save_load(self) -> None:

        with self.subTest("pybind"):
            save_path = os.path.join(self.test_dir, "regex_pybind.pt")

            tokenizer = RegexTokenizer(self.patterns_list)
            torch.save(tokenizer, save_path)
            loaded_tokenizer = torch.load(save_path)
            results = loaded_tokenizer(self.test_sample)
            self.assertEqual(results, self.ref_results)

        with self.subTest("torchscript"):
            save_path = os.path.join(self.test_dir, "regex_torchscript.pt")
            tokenizer = torch.jit.script(RegexTokenizer(self.patterns_list))
            torch.jit.save(tokenizer, save_path)
            loaded_tokenizer = torch.jit.load(save_path)
            results = loaded_tokenizer(self.test_sample)
            self.assertEqual(results, self.ref_results)


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
            with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                self.assertEqual(self.sample_token_ids, masked_tokens)

            # when mask_mask_prob, rand_mask_prob = 0,1 we expect all tokens selected for replacement to be
            # changed to a random token_id
            with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.transforms.MaskTransform.rand_mask_prob", 1.0
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
            with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 1.0), patch(
                "torchtext.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                exp_tokens = torch.tensor([[5, 0, 1, 2, 3], [6, 0, 1, 4, 4]])
                self.assertEqual(exp_tokens, masked_tokens)

        # when mask_prob = 1, we expect all tokens that are not [BOS] or [PAD] to be chosen for replacement
        # (under the default condition that mask_transform.mask_bos=False)
        if test_mask_prob == 1.0:

            # when mask_mask_prob, rand_mask_prob = 0,0 no tokens should change
            with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                self.assertEqual(self.sample_token_ids, masked_tokens)

            # when mask_mask_prob, rand_mask_prob = 0,1 we expect all tokens selected for replacement
            # to be changed to random token_ids. It is possible that the randomly selected token id is the same
            # as the original token id, however we know deterministically that [BOS] and [PAD] tokens
            # in the sequences will remain unchanged.
            with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 0.0), patch(
                "torchtext.transforms.MaskTransform.rand_mask_prob", 1.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                self.assertEqual(masked_tokens[:, 0], 6 * torch.ones_like(masked_tokens[:, 0]))
                self.assertEqual(masked_tokens[1, 3:], 4 * torch.ones_like(masked_tokens[1, 3:]))

            # when mask_mask_prob, rand_mask_prob = 1,0 we expect all tokens selected for replacement to be changed to [MASK]
            with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 1.0), patch(
                "torchtext.transforms.MaskTransform.rand_mask_prob", 0.0
            ):
                masked_tokens, _, _ = mask_transform(self.sample_token_ids)
                exp_tokens = torch.tensor([[6, 5, 5, 5, 5], [6, 5, 5, 4, 4]])
                self.assertEqual(exp_tokens, masked_tokens)

    def test_mask_transform_mask_bos(self) -> None:
        # MaskTransform has boolean parameter mask_bos to indicate whether or not [BOS] tokens
        # should be eligible for replacement. The above tests of MaskTransform are under default value
        # mask_bos = False. Here we test the case where mask_bos = True
        mask_transform = MaskTransform(
            self.vocab_len - 1, self.mask_idx, self.bos_idx, self.pad_idx, mask_bos=True, mask_prob=1.0
        )

        # when mask_mask_prob, rand_mask_prob = 1,0 we expect all tokens selected for replacement to be changed to [MASK]
        with patch("torchtext.transforms.MaskTransform.mask_mask_prob", 1.0), patch(
            "torchtext.transforms.MaskTransform.rand_mask_prob", 0.0
        ):
            masked_tokens, _, _ = mask_transform(self.sample_token_ids)
            exp_tokens = torch.tensor([[5, 5, 5, 5, 5], [5, 5, 5, 4, 4]])
            self.assertEqual(exp_tokens, masked_tokens)
