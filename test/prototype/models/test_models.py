from unittest.mock import patch

from test.common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):
    def test_t5_bundler_build_model(self):
        from torchtext.prototype.models import T5Conf, T5Model, T5Bundle

        # case: user provide encoder checkpoint state dict
        dummy_encoder_conf = T5Conf(
            encoder_only=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
        )
        dummy_t5_encoder = T5Model(dummy_encoder_conf)
        t5_encoder_model = T5Bundle.build_model(config=dummy_encoder_conf, checkpoint=dummy_t5_encoder.state_dict())
        self.assertEqual(t5_encoder_model.state_dict(), dummy_t5_encoder.state_dict())

        # case: user provide encoder-decoder checkpoint state dict
        dummy_t5_conf = T5Conf(
            encoder_only=False,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
        )
        dummy_t5 = T5Model(dummy_t5_conf)
        t5_model = T5Bundle.build_model(config=dummy_t5_conf, checkpoint=dummy_t5.state_dict())
        self.assertEqual(t5_model.state_dict(), dummy_t5.state_dict())

    @patch("logging.Logger.warning")
    def test_t5_bundler_get_model(self, mock):
        from torchtext.prototype.models import T5Conf, T5Bundle

        # encoder-only
        dummy_encoder_conf = T5Conf(
            encoder_only=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
        )
        encoder_bundle = T5Bundle(dummy_encoder_conf)
        encoder_bundle.get_model(load_weights=False, freeze_model=True)
        mock.assert_called_with(
            "The model is not loaded with pre-trained weights. Setting freeze_model to True will hinder model from learning appropriate weights."
        )

        # encoder-decoder
        dummy_t5_conf = T5Conf(
            encoder_only=False,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
        )
        t5_bundle = T5Bundle(dummy_t5_conf)
        t5_bundle.get_model(load_weights=False, freeze_model=True)
        mock.assert_called_with(
            "The model is not loaded with pre-trained weights. Setting freeze_model to True will hinder model from learning appropriate weights."
        )

    def test_t5_bundler_raise_checkpoint(self):
        from torchtext.prototype.models import T5Conf, T5Bundle

        # encoder-only
        with self.assertRaises(TypeError):
            dummy_encoder_conf = T5Conf(
                encoder_only=True,
                vocab_size=10,
                embedding_dim=16,
                ffn_dimension=64,
                num_attention_heads=2,
                num_encoder_layers=2,
            )
            T5Bundle.build_model(
                config=dummy_encoder_conf,
                freeze_model=True,
                checkpoint=1,
            )

        # encoder-decoder
        with self.assertRaises(TypeError):
            dummy_t5_conf = T5Conf(
                encoder_only=False,
                vocab_size=10,
                embedding_dim=16,
                ffn_dimension=64,
                num_attention_heads=2,
                num_encoder_layers=2,
            )
            T5Bundle.build_model(
                config=dummy_t5_conf,
                freeze_model=True,
                checkpoint=1,
            )

    def test_t5_bundler_conf_property(self):
        from torchtext.prototype.models import T5Conf, T5Bundle

        # encoder-only
        dummy_encoder_conf = T5Conf(
            encoder_only=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
        )
        encoder_bundle = T5Bundle(dummy_encoder_conf)
        self.assertTrue(isinstance(encoder_bundle.config, T5Conf))

        # encoder-decoder
        dummy_t5_conf = T5Conf(
            encoder_only=False,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
        )
        t5_bundle = T5Bundle(dummy_t5_conf)
        self.assertTrue(isinstance(t5_bundle.config, T5Conf))
