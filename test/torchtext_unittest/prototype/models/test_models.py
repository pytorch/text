import copy
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torchtext.prototype.models import T5Bundle, T5Conf, T5Model
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):
    def test_t5_bundler_build_model(self) -> None:
        # case: user provides encoder checkpoint state dict
        dummy_encoder_conf = T5Conf(
            encoder_only=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        dummy_t5_encoder = T5Model(dummy_encoder_conf)
        t5_encoder_model = T5Bundle.build_model(
            config=dummy_encoder_conf, checkpoint=dummy_t5_encoder.state_dict()
        )
        self.assertEqual(t5_encoder_model.state_dict(), dummy_t5_encoder.state_dict())

        # case: user provides encoder-decoder checkpoint state dict
        dummy_t5_conf = T5Conf(
            encoder_only=False,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        dummy_t5 = T5Model(dummy_t5_conf)
        t5_model = T5Bundle.build_model(
            config=dummy_t5_conf, checkpoint=dummy_t5.state_dict()
        )
        self.assertEqual(t5_model.state_dict(), dummy_t5.state_dict())

        # case: user provides checkpoint state dict for encoder-decoder with generation
        dummy_t5_generation_conf = T5Conf(
            encoder_only=False,
            linear_head=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        dummy_t5_generation = T5Model(dummy_t5_generation_conf)
        t5_generation_model = T5Bundle.build_model(
            config=dummy_t5_generation_conf, checkpoint=dummy_t5_generation.state_dict()
        )
        self.assertEqual(
            t5_generation_model.state_dict(), dummy_t5_generation.state_dict()
        )

    @patch("logging.Logger.warning")
    def test_t5_bundler_get_model(self, mock):
        # encoder-decoder with generation
        dummy_t5_generation_conf = T5Conf(
            encoder_only=False,
            linear_head=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        t5_generation_bundle = T5Bundle(dummy_t5_generation_conf)
        t5_generation_bundle.get_model(load_weights=False, freeze_model=True)
        mock.assert_called_with(
            "The model is not loaded with pre-trained weights. Setting freeze_model to True will hinder model from learning appropriate weights."
        )

    def test_t5_bundler_raise_checkpoint(self) -> None:
        # encoder-only
        with self.assertRaises(TypeError):
            dummy_encoder_conf = T5Conf(
                encoder_only=True,
                vocab_size=10,
                embedding_dim=16,
                ffn_dimension=64,
                num_attention_heads=2,
                num_encoder_layers=2,
                num_decoder_layers=2,
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
                num_decoder_layers=2,
            )
            T5Bundle.build_model(
                config=dummy_t5_conf,
                freeze_model=True,
                checkpoint=1,
            )

        # encoder-decoder with generation
        with self.assertRaises(TypeError):
            dummy_t5_generation_conf = T5Conf(
                encoder_only=False,
                linear_head=True,
                vocab_size=10,
                embedding_dim=16,
                ffn_dimension=64,
                num_attention_heads=2,
                num_encoder_layers=2,
                num_decoder_layers=2,
            )
            T5Bundle.build_model(
                config=dummy_t5_generation_conf,
                freeze_model=True,
                checkpoint=1,
            )

    def test_t5_bundler_conf_property(self) -> None:
        dummy_t5_conf = T5Conf(
            encoder_only=False,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        t5_bundle = T5Bundle(dummy_t5_conf)
        self.assertTrue(isinstance(t5_bundle.config, T5Conf))

    def test_t5_bundler_train(self) -> None:
        from torch.optim import SGD

        def _train(model):
            optim = SGD(model.parameters(), lr=1)
            model_input = torch.tensor([[1, 2, 3, 4, 5]])
            target = torch.tensor([1])
            output = model(model_input)["decoder_output"]
            logits = F.log_softmax(output[:, -1], dim=-1)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optim.step()

        dummy_conf = T5Conf(
            encoder_only=False,
            linear_head=True,
            vocab_size=10,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            training=True,
        )
        dummy_model = T5Model(dummy_conf)
        model = T5Bundle.build_model(
            config=dummy_conf,
            freeze_model=False,
            checkpoint=dummy_model.state_dict(),
        )
        current_state_dict = copy.deepcopy(model.state_dict())

        _train(model)
        self.assertNotEqual(model.state_dict(), current_state_dict)

    def test_t5_model_forward_with_encoder_mask_encoder_only(self) -> None:
        dummy_conf = T5Conf(
            encoder_only=True,
            linear_head=True,
            vocab_size=100,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            training=False,
        )
        dummy_model = T5Model(dummy_conf)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
        mask = tokens.eq(0)

        with torch.no_grad():
            output_with_mask = dummy_model(
                encoder_tokens=tokens, encoder_padding_mask=mask
            )
            output_no_mask = dummy_model(tokens)

        torch.testing.assert_close(
            output_with_mask["encoder_output"],
            output_no_mask["encoder_output"],
            atol=1e-04,
            rtol=2.5e-06,
        )

    def test_t5_model_forward_with_encoder_mask_encoder_decoder(self) -> None:
        dummy_conf = T5Conf(
            encoder_only=False,
            linear_head=True,
            vocab_size=100,
            embedding_dim=16,
            ffn_dimension=64,
            num_attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            training=False,
        )
        dummy_model = T5Model(dummy_conf)
        enc_tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
        dec_tokens = torch.tensor([[6, 7, 8, 9, 10, 11, 0, 0, 0, 0]])
        enc_mask = enc_tokens.eq(0)
        dec_mask = dec_tokens.eq(0)

        with torch.no_grad():
            output_with_mask = dummy_model(
                encoder_tokens=enc_tokens,
                encoder_padding_mask=enc_mask,
                decoder_tokens=dec_tokens,
                decoder_padding_mask=dec_mask,
            )
            output_no_mask = dummy_model(
                encoder_tokens=enc_tokens, decoder_tokens=dec_tokens
            )

        torch.testing.assert_close(
            output_with_mask["decoder_output"],
            output_no_mask["decoder_output"],
            atol=1e-04,
            rtol=2.5e-06,
        )
