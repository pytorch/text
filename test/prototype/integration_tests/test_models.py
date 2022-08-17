import torch
from parameterized import parameterized
from test.common.assets import get_asset_path
from test.common.parameterized_utils import nested_params
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.prototype.models import (
    T5_BASE_ENCODER,
    T5_BASE,
    T5_BASE_GENERATION,
    T5_SMALL_ENCODER,
    T5_SMALL,
    T5_SMALL_GENERATION,
    T5_LARGE_ENCODER,
    T5_LARGE,
    T5_LARGE_GENERATION,
    T5Conf,
    T5Transform,
)
from torchtext.prototype.models.t5.wrapper import T5Wrapper


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


class TestT5(TorchtextTestCase):
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
            actual = model(model_input)["encoder_output"]
        else:
            actual = model(model_input)["decoder_output"]

        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected, atol=1e-04, rtol=2.5e-06)

    @nested_params(["base", "small", "large"], ["encoder", "model", "generation"], ["jit", "not_jit"])
    def test_t5_encoder_model(self, configuration, type, name) -> None:
        expected_asset_name = f"t5.{configuration}.{type}.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        is_jit = name == "jit"
        t5_model = BUNDLERS[configuration + "_" + type]
        self._t5_model(is_jit=is_jit, t5_model=t5_model, expected_asset_name=expected_asset_name, test_text=test_text)

    @nested_params(["base", "small", "large"], ["jit", "not_jit"])
    def test_t5_wrapper(self, configuration, name) -> None:
        test_text = ["translate English to French: I want to eat pizza for dinner."]
        if configuration == "small":
            expected_text = ["Je veux manger la pizza pour le dîner."]
        else:
            expected_text = ["Je veux manger de la pizza pour le dîner."]
        beam_size = 3
        max_seq_len = 512
        model = T5Wrapper(configuration=configuration)
        if name == "jit":
            model = torch.jit.script(model)

        output_text = model(test_text, beam_size, max_seq_len)
        self.assertEqual(output_text, expected_text)

    @parameterized.expand(["jit", "not_jit"])
    def test_t5_wrapper_checkpoint(self, name) -> None:
        test_text = ["translate English to French: I want to eat pizza for dinner."]
        expected_text = ["Je veux manger de la pizza pour le dîner."]
        beam_size = 3
        max_seq_len = 512
        config = T5Conf(encoder_only=False, linear_head=True)
        transform = T5Transform(
            "https://download.pytorch.org/models/text/t5_tokenizer_base.model",
            max_seq_len=512,
            eos_idx=1,
            padding_idx=0,
        )
        model = T5Wrapper(
            checkpoint="https://download.pytorch.org/models/text/t5.base.generation.pt",
            t5_config=config,
            transform=transform,
            freeze_model=True,
            strict=True,
        )
        if name == "jit":
            model = torch.jit.script(model)

        output_text = model(test_text, beam_size, max_seq_len)
        self.assertEqual(output_text, expected_text)
