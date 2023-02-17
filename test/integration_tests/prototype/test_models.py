import tempfile

import pytest  # noqa: F401
import torch
from parameterized import parameterized, parameterized_class
from torchtext.prototype.models import (
    T5_BASE,
    T5_BASE_ENCODER,
    T5_BASE_GENERATION,
    T5_LARGE,
    T5_LARGE_ENCODER,
    T5_LARGE_GENERATION,
    T5_SMALL,
    T5_SMALL_ENCODER,
    T5_SMALL_GENERATION,
    T5Conf,
    T5Transform,
)
from torchtext.prototype.models.t5.bundler import T5Bundle
from torchtext_unittest.common.assets import get_asset_path
from torchtext_unittest.common.parameterized_utils import nested_params
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Model

BUNDLERS = {
    "base_model": T5_BASE,
    "base_encoder": T5_BASE_ENCODER,
    "base_generation": T5_BASE_GENERATION,
    "small_model": T5_SMALL,
    "small_encoder": T5_SMALL_ENCODER,
    "small_generation": T5_SMALL_GENERATION,
    "large_model": T5_LARGE,
    "large_encoder": T5_LARGE_ENCODER,
    "large_generation": T5_LARGE_GENERATION,
}


@parameterized_class(
    ("model_name",),
    [
        ("base_model",),
        ("base_encoder",),
        ("base_generation",),
        ("small_model",),
        ("small_encoder",),
        ("small_generation",),
        ("large_model",),
        ("large_encoder",),
        ("large_generation",),
    ],
)
class TestT5Model(TorchtextTestCase):
    def _t5_model(self, is_jit, t5_model, expected_asset_name, test_text):
        """Verify that pre-trained T5 models in torchtext produce
        the same output as the HuggingFace reference implementation.
        """
        expected_asset_path = get_asset_path(expected_asset_name)
        transform = t5_model.transform()
        model = t5_model.get_model()
        model = model.eval()

        if is_jit:
            transform = torch.jit.script(transform)
            model = torch.jit.script(model)

        model_input = transform(test_text)
        if model.encoder_only:
            actual = model(encoder_tokens=model_input)["encoder_output"]
            if not is_jit:
                self._t5_get_encoder(model, model_input, actual)
        else:
            actual = model(encoder_tokens=model_input)["decoder_output"]

        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected, atol=1e-04, rtol=2.5e-06)

    def _t5_get_encoder(self, model, model_input, encoder_output):
        encoder = model.get_encoder()
        # Need to set the tgt_key_padding_mask to ensure the same results
        encoder_padding_mask = model_input.eq(model.padding_idx)
        output_from_get_encoder = encoder(tgt=model_input, tgt_key_padding_mask=encoder_padding_mask)["encoder_output"]
        assert torch.all(output_from_get_encoder.eq(encoder_output))

    @nested_params(["jit", "not_jit"])
    def test_t5_model(self, name) -> None:
        configuration, type = self.model_name.split("_")

        expected_asset_name = f"t5.{configuration}.{type}.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        is_jit = name == "jit"
        t5_model = BUNDLERS[configuration + "_" + type]
        self._t5_model(is_jit=is_jit, t5_model=t5_model, expected_asset_name=expected_asset_name, test_text=test_text)


class TestLoadFromHFCheckpoints(TorchtextTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.encoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]])
        self.encoder_padding_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
        self.decoder_input_ids = torch.tensor([[7, 8, 9, 0, 0, 0], [10, 11, 12, 0, 0, 0]])
        self.decoder_padding_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])

    def check_outputs_of_models(self, our_output, hf_output, config, encoder_only) -> None:
        # check that encoder layers match
        for i in range(config.num_encoder_layers + 1):
            if i < config.num_encoder_layers:
                hf_output_sa = hf_output.attentions[i] if encoder_only else hf_output.encoder_attentions[i]
                # self-attention scores
                assert torch.equal(
                    our_output["encoder_sa_scores"][i], hf_output_sa
                ), f"Mismatched self-attention scores for encoder layer {i}"
            hf_output_hs = hf_output.hidden_states[i] if encoder_only else hf_output.encoder_hidden_states[i]
            # encoder hidden states
            assert torch.equal(
                our_output["encoder_hidden_states"][i], hf_output_hs
            ), f"Mismatched hidden states for encoder layer {i}"

        if not encoder_only:
            # check that decoder layers match
            for i in range(config.num_decoder_layers + 1):
                if i < config.num_encoder_layers:
                    # self-attention scores
                    assert torch.equal(
                        our_output["decoder_sa_scores"][i], hf_output.decoder_attentions[i]
                    ), f"Mismatched self-attention scores for decoder layer {i}"
                    # cross-attention scores
                    assert torch.equal(
                        our_output["decoder_ca_scores"][i], hf_output.cross_attentions[i]
                    ), f"Mismatched cross-attention scores for decoder layer {i}"
                # decoder hidden states
                assert torch.equal(
                    our_output["decoder_hidden_states"][i], hf_output.decoder_hidden_states[i]
                ), f"Mismatched hidden states for decoder layer {i}"

    def test_t5_bundler_load_hf_ckpt_pretrained_encoder_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = f"{tmp_dir}/hf_t5_small_enc"

            t5_small_enc = T5EncoderModel.from_pretrained("t5-small")
            t5_small_enc.save_pretrained(model_path)

            our_encoder = T5Bundle.build_model_from_huggingface_ckpt(model_path, encoder_only=True)

            hf_output = t5_small_enc(
                input_ids=self.encoder_input_ids,
                attention_mask=self.encoder_padding_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

            our_output = our_encoder(self.encoder_input_ids)

            self.check_outputs_of_models(our_output, hf_output, our_encoder.config, True)

    def test_t5_bundler_load_hf_ckpt_pretrained_encoder_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = f"{tmp_dir}/hf_t5_small"

            t5_small = T5Model.from_pretrained("t5-small")
            t5_small.save_pretrained(model_path)

            our_t5 = T5Bundle.build_model_from_huggingface_ckpt(model_path)

            hf_output = t5_small(
                input_ids=self.encoder_input_ids,
                decoder_input_ids=self.decoder_input_ids,
                attention_mask=self.encoder_padding_mask,
                decoder_attention_mask=self.decoder_padding_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

            our_output = our_t5(self.encoder_input_ids, self.decoder_input_ids)

            self.check_outputs_of_models(our_output, hf_output, our_t5.config, False)

    def test_t5_bundler_load_hf_ckpt_pretrained_encoder_decoder_with_gen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = f"{tmp_dir}/hf_t5_small_gen"

            t5_small_gen = T5ForConditionalGeneration.from_pretrained("t5-small")
            t5_small_gen.save_pretrained(model_path)

            our_t5 = T5Bundle.build_model_from_huggingface_ckpt(model_path)

            hf_output = t5_small_gen(
                input_ids=self.encoder_input_ids,
                decoder_input_ids=self.decoder_input_ids,
                attention_mask=self.encoder_padding_mask,
                decoder_attention_mask=self.decoder_padding_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

            our_output = our_t5(self.encoder_input_ids, self.decoder_input_ids)

            self.check_outputs_of_models(our_output, hf_output, our_t5.config, False)

    def test_flan_t5_bundler_load_hf_ckpt_pretrained_encoder_decoder(self) -> None:
        # TODO(joecummings): Download FLAN-T5 chkpts and test here
        pass
