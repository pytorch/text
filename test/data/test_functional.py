import os
import shutil
import tempfile
import uuid
import unittest
import regex as re

import torch
from torchtext.data.functional import (
    generate_sp_model,
    load_sp_model,
    sentencepiece_numericalizer,
    sentencepiece_tokenizer,
    custom_replace,
    simple_space_split,
    gpt2_bpe_pre_tokenizer,
)

from ..common.torchtext_test_case import TorchtextTestCase
from ..common.assets import get_asset_path


class TestFunctional(TorchtextTestCase):
    def test_generate_sp_model(self):
        """
        Test the function to train a sentencepiece tokenizer.
        """

        asset_name = 'text_normalization_ag_news_test.csv'
        asset_path = get_asset_path(asset_name)
        # We use temporary directory for two reasons:
        # 1. buck (fb internal) generates test environment which contains ',' in its path.
        #    SentencePieceTrainer considers such path as comma-delimited file list.
        #    So as workaround we copy the asset data to temporary directory and load it from there.
        # 2. when fb infra performs stress tests, multiple instances of this test run.
        #    The name of the generated models have to be unique and they need to be cleaned up.
        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)

            model_prefix = os.path.join(dir_name, f'spm_user_{uuid.uuid4()}')
            model_file = f'{model_prefix}.model'
            generate_sp_model(data_path, vocab_size=23456, model_prefix=model_prefix)
            sp_model = load_sp_model(model_file)
            self.assertEqual(sp_model.GetPieceSize(), 23456)

    def test_sentencepiece_numericalizer(self):
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = get_asset_path('spm_example.model')
        sp_model = load_sp_model(model_path)
        self.assertEqual(sp_model.GetPieceSize(), 20000)
        spm_generator = sentencepiece_numericalizer(sp_model)

        ref_results = [15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                       144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

    def test_sentencepiece_tokenizer(self):
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = get_asset_path('spm_example.model')
        sp_model = load_sp_model(open(model_path, 'rb'))
        self.assertEqual(sp_model.GetPieceSize(), 20000)
        spm_generator = sentencepiece_tokenizer(sp_model)

        ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                       '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                       '\u2581to', 'ken', 'izer', '\u2581and',
                       '\u2581de', 'to', 'ken', 'izer']

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

    def test_sentencepiece_unsupported_input_type(self):
        with self.assertRaisesRegex(
            TypeError,
            'Unsupported type for spm argument: dict. '
            'Supported types are: str, io.BufferedReader'
        ):
            load_sp_model(dict())

    def test_custom_replace(self):
        custom_replace_transform = custom_replace([(r'S', 's'), (r'\s+', ' ')])
        test_sample = ['test     cuStom   replace', 'with   uSer   instruction']
        ref_results = ['test custom replace', 'with user instruction']
        self.assertEqual(list(custom_replace_transform(test_sample)),
                         ref_results)

    def test_simple_space_split(self):
        test_sample = ['test simple space split function']
        ref_results = ['test', 'simple', 'space', 'split', 'function']
        self.assertEqual(list(simple_space_split(test_sample))[0],
                         ref_results)

    def test_gpt2_bpe_pre_tokenizer(self):
        # Regex pattern for GPT-2 BPE which includes the negative lookahead
        # Reference: https://github.com/pytorch/fairseq/blob/main/fairseq/data/encoders/gpt2_bpe_utils.py#L69
        gpt2_bpe_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        test_cases = [
            # test spaces
            "Lorem ipsum dolor sit amet.",
            "Lorem ipsum        dolor sit amet.",
            "Lorem ipsum        dolor sit amet.  ",
            "Lorem ipsum dolor sit amet   ",
            "Lorem\x0d\x0dipsum dolor sit amet\r\r",
            "Lorem ipsum\x20dolor sit amet",
            "Lorem ipsum\x20\x20\x20dolor sit amet",
            "Lorem ipsum\x20\x20 dolor sit amet",
            # test tabs
            "Lorem ipsum dolor sit \t\t\t amet.",
            "Lorem ipsum dolor sit \t\t\t\tamet.",
            "Lorem ipsum dolor sit \x09\x09amet.",
            "Lorem ipsum dolor sit \x09\x09 amet.",
            "Lorem ipsum dolor sit \x09\x09 amet.   ",
            "Lorem ipsum dolor sit \t   \tamet.",
            "Lorem ipsum dolor sit amet   \t",
            "Lorem ipsum\tdolor sit amet",
            # test carriage returns
            "Lorem ipsum\r\r dolor sit amet",
            "Lorem ipsum\r\r dolor sit amet\r\r",
            "Lorem ipsum \x0d\x0ddolor sit amet.",
            "Lorem ipsum\x0ddolor sit amet.",
            "Lorem ipsum\x0d\x0d dolor sit amet.",
            "Lorem ipsum\x0d\x0d dolor sit amet.\x0d",
            # test form feeds
            "Lorem ipsum\f\fdolor sit amet\f",
            "Lorem ipsum\f\f dolor sit amet\f ",
            "Lorem ipsum\x0c\x0c dolor sit amet",
            "Lorem \x0c\x0c\x0c\x0cipsum dolor sit amet",
            # test vertical tabs
            "Lorem ipsum dolor sit\vamet.",
            "Lorem ipsum dolor sit\v\vamet.",
            "Lorem ipsum dolor sit\v\v amet.",
            "Lorem ipsum dolor sit\v\v amet.  \v",
            "Lorem ipsum dolor sit\x0b\x0b amet.  \v ",
            "Lorem ipsum dolor sit\x0bamet.",
            "Lorem ipsum dolor sit\x0b\x0bamet.",
            "Lorem ipsum dolor sit\x0b\x0b amet.",
        ]
        for t in test_cases:
            self.assertEqual(re.findall(gpt2_bpe_pattern, t), gpt2_bpe_pre_tokenizer(t))


class ScriptableSP(torch.jit.ScriptModule):
    def __init__(self, model_path):
        super().__init__()
        self.spm = load_sp_model(model_path)

    @torch.jit.script_method
    def encode(self, input: str):
        return self.spm.Encode(input)

    @torch.jit.script_method
    def encode_as_ids(self, input: str):
        return self.spm.EncodeAsIds(input)

    @torch.jit.script_method
    def encode_as_pieces(self, input: str):
        return self.spm.EncodeAsPieces(input)


class TestScriptableSP(unittest.TestCase):
    def setUp(self):
        model_path = get_asset_path('spm_example.model')
        with tempfile.TemporaryDirectory() as dir_name:
            jit_model_path = os.path.join(dir_name, 'spm_example.model')
            torch.jit.script(ScriptableSP(model_path)).save(jit_model_path)
            self.model = torch.jit.load(jit_model_path)

    def test_encode(self):
        input = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        expected = [
            '▁Sent', 'ence', 'P', 'ie', 'ce', '▁is',
            '▁an', '▁un', 'super', 'vis', 'ed', '▁text',
            '▁to', 'ken', 'izer', '▁and',
            '▁de', 'to', 'ken', 'izer',
        ]
        output = self.model.encode(input)
        self.assertEqual(expected, output)

    def test_encode_as_ids(self):
        input = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        expected = [
            15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
            144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]
        output = self.model.encode_as_ids(input)
        self.assertEqual(expected, output)

    def test_encode_as_pieces(self):
        input = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        expected = [
            '\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
            '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
            '\u2581to', 'ken', 'izer', '\u2581and',
            '\u2581de', 'to', 'ken', 'izer',
        ]
        output = self.model.encode_as_pieces(input)
        self.assertEqual(expected, output)
