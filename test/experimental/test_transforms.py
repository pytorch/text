import torch
from test.common.torchtext_test_case import TorchtextTestCase
from test.common.assets import get_asset_path
from torchtext.experimental.transforms import (
    VectorTransform,
    VocabTransform,
)
from torchtext.experimental.vocab import vocab_from_file_object
from torchtext.experimental.vectors import FastText


class TestTransforms(TorchtextTestCase):
    def test_vocab_transform(self):
        asset_name = 'vocab_test2.txt'
        asset_path = get_asset_path(asset_name)
        f = open(asset_path, 'r')
        vocab_transform = VocabTransform(vocab_from_file_object(f))
        self.assertEqual(vocab_transform(['of', 'that', 'new']), [7, 18, 24])
        jit_vocab_transform = torch.jit.script(vocab_transform.to_ivalue())
        self.assertEqual(jit_vocab_transform(['of', 'that', 'new']), [7, 18, 24])

    def test_vector_transform(self):
        asset_name = 'wiki.en.vec'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)
            vectors_transform = VectorTransform(FastText(root=dir_name, validate_file=False))
            jit_vectors_transform = torch.jit.script(vectors_transform.to_ivalue())

            # The first 3 entries in each vector.
            expected_fasttext_simple_en = {
                'the': [-0.065334, -0.093031, -0.017571],
                'world': [-0.32423, -0.098845, -0.0073467],
            }

            for word in expected_fasttext_simple_en.keys():
                self.assertEqual(vectors_transform[word][:3], expected_fasttext_simple_en[word])
                self.assertEqual(jit_vectors_transform[word][:3], expected_fasttext_simple_en[word])
