import os
import unittest
import tempfile

import sentencepiece as spm
import torch
from torchtext.data.functional import (
    generate_sp_model,
    load_sp_model,
    sentencepiece_numericalizer,
    sentencepiece_tokenizer,
    custom_replace,
    simple_space_split,
)

from ..common.torchtext_test_case import TorchtextTestCase


class TestFunctional(TorchtextTestCase):
    def test_generate_sp_model(self):
        # Test the function to train a sentencepiece tokenizer

        data_path = 'test/asset/text_normalization_ag_news_test.csv'
        generate_sp_model(data_path,
                          vocab_size=23456,
                          model_prefix='spm_user')

        sp_user = spm.SentencePieceProcessor()
        sp_user.Load('spm_user.model')

        self.assertEqual(len(sp_user), 23456)

        if os.path.isfile('spm_user.model'):
            os.remove('spm_user.model')
        if os.path.isfile('spm_user.vocab'):
            os.remove('spm_user.vocab')

    def test_sentencepiece_numericalizer(self):
        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_model = load_sp_model(model_path)
        self.assertEqual(sp_model.GetPieceSize(), 20000)
        spm_generator = sentencepiece_numericalizer(sp_model)

        ref_results = [15340, 4286, 981, 1207, 1681, 17, 84, 684, 8896, 5366,
                       144, 3689, 9, 5602, 12114, 6, 560, 649, 5602, 12114]

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

    def test_sentencepiece_tokenizer(self):

        test_sample = 'SentencePiece is an unsupervised text tokenizer and detokenizer'
        model_path = 'test/asset/spm_example.model'
        sp_model = load_sp_model(model_path)
        self.assertEqual(sp_model.GetPieceSize(), 20000)
        spm_generator = sentencepiece_tokenizer(sp_model)

        ref_results = ['\u2581Sent', 'ence', 'P', 'ie', 'ce', '\u2581is',
                       '\u2581an', '\u2581un', 'super', 'vis', 'ed', '\u2581text',
                       '\u2581to', 'ken', 'izer', '\u2581and',
                       '\u2581de', 'to', 'ken', 'izer']

        self.assertEqual(list(spm_generator([test_sample]))[0],
                         ref_results)

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
        model_path = 'test/asset/spm_example.model'
        with tempfile.NamedTemporaryFile() as file:
            torch.jit.script(ScriptableSP(model_path)).save(file.name)
            self.model = torch.jit.load(file.name)

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
