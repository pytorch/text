import copy
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torchtext.prototype.models.t5.bundler import T5Bundle
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase

def test_model(our_output, hf_output, config, encoder_only):
    
    if encoder_only:
        #check that encoder layers match
        for i in range(config.num_encoder_layers + 1):
            if i < config.num_encoder_layers:
                #self-attention scores
                assert torch.equal(
                    our_output['encoder_sa_scores'][i],
                    hf_output.attentions[i]
                ), f"Mismatched self-attention scores for encoder layer {i}"
            #encoder hidden states
            assert torch.equal(
                our_output['encoder_hidden_states'][i],
                hf_output.hidden_states[i]
            ), f"Mismatched hidden states for encoder layer {i}"
    
    else:
        #check that encoder layers match
        for i in range(config.num_encoder_layers + 1):
            if i < config.num_encoder_layers:
                #self-attention scores
                assert torch.equal(
                    our_output['encoder_sa_scores'][i],
                    hf_output.encoder_attentions[i]
                ), f"Mismatched self-attention scores for encoder layer {i}"
            #encoder hidden states
            assert torch.equal(
                our_output['encoder_hidden_states'][i],
                hf_output.encoder_hidden_states[i]
            ), f"Mismatched hidden states for encoder layer {i}"

        # check that decoder layers match
        for i in range(config.num_decoder_layers + 1):
            if i < config.num_encoder_layers:
                #self-attention scores
                assert torch.equal(
                    our_output['decoder_sa_scores'][i],
                    hf_output.decoder_attentions[i]
                ), f"Mismatched self-attention scores for decoder layer {i}"
                #cross-attention scores
                assert torch.equal(
                    our_output['decoder_ca_scores'][i],
                    hf_output.cross_attentions[i]
                ), f"Mismatched cross-attention scores for decoder layer {i}"
            #decoder hidden states
            assert torch.equal(
                our_output['decoder_hidden_states'][i],
                hf_output.decoder_hidden_states[i]
            ), f"Mismatched hidden states for decoder layer {i}"


class TestModels(TorchtextTestCase):
    def test_t5_bundler_build_model(self) -> None:
        from torchtext.prototype.models import T5Conf, T5Model, T5Bundle

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
        t5_encoder_model = T5Bundle.build_model(config=dummy_encoder_conf, checkpoint=dummy_t5_encoder.state_dict())
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
        t5_model = T5Bundle.build_model(config=dummy_t5_conf, checkpoint=dummy_t5.state_dict())
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
        self.assertEqual(t5_generation_model.state_dict(), dummy_t5_generation.state_dict())

    @patch("logging.Logger.warning")
    def test_t5_bundler_get_model(self, mock):
        from torchtext.prototype.models import T5Conf, T5Bundle

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
        from torchtext.prototype.models import T5Conf, T5Bundle

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
        from torchtext.prototype.models import T5Conf, T5Model, T5Bundle

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

    def test_t5_bundler_load_hf_ckpt_pretrained_encoder_only(self) -> None:
        from transformers import T5EncoderModel
        hf_encoder = T5EncoderModel.from_pretrained("t5-small")

        our_encoder = T5Bundle.build_model_from_huggingface_ckpt("t5-small")

        encoder_input_ids = torch.tensor([[1,2,3,4,5,6],[7,8,9,0,0,0]])
        encoder_padding_mask = torch.tensor([[1,1,1,1,1,1],[1,1,1,0,0,0]])

        hf_output = hf_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_padding_mask,
            output_hidden_states=True,
            output_attentions=True
        )

        our_output = our_encoder(encoder_input_ids)

        test_model(our_output, hf_output, our_encoder.config, encoder_only=True)