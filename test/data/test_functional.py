from ..common.torchtext_test_case import TorchtextTestCase
import sentencepiece as spm


class TestUtils(TorchtextTestCase):
    def test_generate_sp_tokenizer(self):
        # Test the function to train a sentencepiece tokenizer
        from torchtext.data.functional import generate_sp_tokenizer
        import os

        data_path = 'test/asset/text_normalization_ag_news_test.csv'
        generate_sp_tokenizer(data_path,
                              vocab_size=23456,
                              model_prefix='spm_user')

        sp_user = spm.SentencePieceProcessor()
        sp_user.Load('spm_user.model')

        self.assertEqual(len(sp_user), 23456)

        if os.path.isfile('spm_user.model'):
            os.remove('spm_user.model')
        if os.path.isfile('spm_user.vocab'):
            os.remove('spm_user.vocab')
