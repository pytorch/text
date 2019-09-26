from ..common.torchtext_test_case import TorchtextTestCase
from torchtext.data.transforms import simple_tokenizer


class TestTransforms(TorchtextTestCase):
    def test_simple_tokenizer(self):
        test_sample = 'You can now install TorchText using pip!'
        tokenizer = simple_tokenizer()
        ref_results = ['You', 'can', 'now', 'install', 'TorchText', 'using', 'pip!']
        self.assertEqual(next(tokenizer([test_sample])), ref_results)
