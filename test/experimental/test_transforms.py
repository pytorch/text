import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.transforms import (
    basic_english_normalize,
    VectorTransform,
    VocabTransform,
    PadTransform,
    sentencepiece_processor,
    sentencepiece_tokenizer,
    TextSequentialTransforms,
    pretrained_sp_model,
)
from torchtext.utils import download_from_url
from torchtext.experimental.vocab import vocab_from_file
from torchtext.experimental.vectors import FastText
import shutil
import tempfile
import os


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

    def test_builtin_pretrained_sentencepiece_processor(self):
        sp_model_path = download_from_url(pretrained_sp_model['text_unigram_25000'])
        spm_tokenizer = sentencepiece_tokenizer(sp_model_path)
        _path = os.path.join(self.project_root, '.data', 'text_unigram_25000.model')
        os.remove(_path)
        test_sample = 'the pretrained spm model names'
        ref_results = ['\u2581the', '\u2581pre', 'trained', '\u2581sp', 'm', '\u2581model', '\u2581names']
        self.assertEqual(spm_tokenizer(test_sample), ref_results)

        sp_model_path = download_from_url(pretrained_sp_model['text_bpe_25000'])
        spm_transform = sentencepiece_processor(sp_model_path)
        _path = os.path.join(self.project_root, '.data', 'text_bpe_25000.model')
        os.remove(_path)
        test_sample = 'the pretrained spm model names'
        ref_results = [13, 1465, 12824, 304, 24935, 5771, 3776]
        self.assertEqual(spm_transform(test_sample), ref_results)

    def test_vocab_transform(self):
        asset_name = 'vocab_test2.txt'
        asset_path = get_asset_path(asset_name)
        with open(asset_path, 'r') as f:
            vocab_transform = VocabTransform(vocab_from_file(f))
            self.assertEqual(vocab_transform(['of', 'that', 'new']),
                             [7, 18, 24])
            jit_vocab_transform = torch.jit.script(vocab_transform.to_ivalue())
            self.assertEqual(jit_vocab_transform(['of', 'that', 'new', 'that']),
                             [7, 18, 24, 18])

    def test_text_sequential_transform(self):
        asset_name = 'vocab_test2.txt'
        asset_path = get_asset_path(asset_name)
        with open(asset_path, 'r') as f:
            pipeline = TextSequentialTransforms(basic_english_normalize(), vocab_from_file(f))
            jit_pipeline = torch.jit.script(pipeline.to_ivalue())
            self.assertEqual(pipeline('of that new'), [7, 18, 24])
            self.assertEqual(jit_pipeline('of that new'), [7, 18, 24])

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

    def test_padding_func(self):
        pad_id = 2
        pad_transform = PadTransform(pad_id)
        # Test torch.int64
        seq_batch = [torch.tensor([5, 4, 5, 6, 7]), torch.tensor([1, 3]), torch.tensor([7, 5, 8])]
        pad_seq, padding_mask = pad_transform(seq_batch)
        expected_pad_seq = torch.tensor([[5, 4, 5, 6, 7], [1, 3, 2, 2, 2], [7, 5, 8, 2, 2]], dtype=torch.long)
        expected_padding_mask = torch.tensor([[False, False, False, False, False],
                                              [False, False, True, True, True],
                                              [False, False, False, True, True]])
        self.assertEqual(pad_seq, expected_pad_seq)
        self.assertEqual(pad_seq.dtype, torch.long)
        self.assertEqual(padding_mask, expected_padding_mask)
        jit_pad_transform = torch.jit.script(pad_transform)
        jit_pad_seq, jit_padding_mask = jit_pad_transform(seq_batch)
        self.assertEqual(jit_pad_seq, expected_pad_seq)
        self.assertEqual(jit_pad_seq.dtype, torch.long)
        self.assertEqual(jit_padding_mask, expected_padding_mask)

        # Test torch.float32
        seq_batch = [torch.tensor([5.0, 4.0, 5.0, 6.0, 7.0]), torch.tensor([1.0, 3.0]), torch.tensor([7.0, 5.0, 8.0])]
        pad_seq, padding_mask = pad_transform(seq_batch)
        expected_pad_seq = expected_pad_seq.to(torch.float32)
        self.assertEqual(pad_seq, expected_pad_seq)
        self.assertEqual(pad_seq.dtype, torch.float32)
        self.assertEqual(padding_mask, expected_padding_mask)
        jit_pad_seq, jit_padding_mask = jit_pad_transform(seq_batch)
        self.assertEqual(jit_pad_seq, expected_pad_seq)
        self.assertEqual(jit_pad_seq.dtype, torch.float32)
        self.assertEqual(jit_padding_mask, expected_padding_mask)
