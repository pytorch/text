import os
import unittest
import sys
import uuid
import shutil
import tempfile

import sentencepiece as spm
import torch
import torchtext.data as data
from torchtext.data.functional import (
    generate_sp_model,
    load_sp_model,
    sentencepiece_numericalizer,
    sentencepiece_tokenizer,
    custom_replace,
    simple_space_split,
)
from torchtext.experimental.transforms import (
    BasicEnglishNormalize,
    RegexTokenizer
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

            sp_user = spm.SentencePieceProcessor()
            sp_user.Load(model_file)

            self.assertEqual(len(sp_user), 23456)

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
        sp_model = load_sp_model(model_path)
        self.assertEqual(sp_model.GetPieceSize(), 20000)
        spm_generator = sentencepiece_tokenizer(sp_model)

        ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                       '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                       '\u2581to', 'ken', 'izer', '\u2581and',
                       '\u2581de', 'to', 'ken', 'izer']

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

    def test_BasicEnglishNormalize(self):
        test_sample = '\'".<br />,()!?;:   Basic English Normalization for a Line of Text   \'".<br />,()!?;:'
        ref_results = ["'", '.', ',', '(', ')', '!', '?', 'basic', 'english', 'normalization',
                       'for', 'a', 'line', 'of', 'text', "'", '.', ',', '(', ')', '!', '?']

        basic_english_normalize = BasicEnglishNormalize()
        experimental_tokens = basic_english_normalize.forward(test_sample)

        basic_english_tokenizer = data.get_tokenizer("basic_english")
        tokens = basic_english_tokenizer(test_sample)

        self.assertEqual(experimental_tokens, ref_results)
        self.assertEqual(experimental_tokens, tokens)

    def test_RegexTokenizer(self):
        test_sample = '\'".<br />,()!?;:   Basic English Normalization for a Line of Text   \'".<br />,()!?;:'
        ref_results = ["'", '.', ',', '(', ')', '!', '?', 'basic', 'english', 'normalization',
                       'for', 'a', 'line', 'of', 'text', "'", '.', ',', '(', ')', '!', '?']
        patterns_list = [
            (r'\'', ' \'  '),
            (r'\"', ''),
            (r'\.', ' . '),
            (r'<br \/>', ' '),
            (r',', ' , '),
            (r'\(', ' ( '),
            (r'\)', ' ) '),
            (r'\!', ' ! '),
            (r'\?', ' ? '),
            (r'\;', ' '),
            (r'\:', ' '),
            (r'\s+', ' ')]

        regex_tokenizer = RegexTokenizer(patterns_list)
        tokens = regex_tokenizer.forward(test_sample)

        self.assertEqual(tokens, ref_results)

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
        with tempfile.NamedTemporaryFile() as file:
            torch.jit.script(ScriptableSP(model_path)).save(file.name)
            self.model = torch.jit.load(file.name)

    @unittest.skipIf(sys.platform == "win32", "FIXME: tempfile could not be opened twice on Windows")
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

    @unittest.skipIf(sys.platform == "win32", "FIXME: tempfile could not be opened twice on Windows")
    def test_encode_as_ids(self):
        input = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        expected = [
            15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
            144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]
        output = self.model.encode_as_ids(input)
        self.assertEqual(expected, output)

    @unittest.skipIf(sys.platform == "win32", "FIXME: tempfile could not be opened twice on Windows")
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
