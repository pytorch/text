import torch
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
)


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

    @nested_params(["base", "small", "large"], ["jit", "not_jit"])
    def test_t5_encoder_model(self, configuration, name) -> None:
        expected_asset_name = f"t5.{configuration}.encoder.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        is_jit = name == "jit"
        if configuration == "base":
            t5_model = T5_BASE_ENCODER
        elif configuration == "small":
            t5_model = T5_SMALL_ENCODER
        elif configuration == "large":
            t5_model = T5_LARGE_ENCODER

        self._t5_model(is_jit=is_jit, t5_model=t5_model, expected_asset_name=expected_asset_name, test_text=test_text)

    @nested_params(["base", "small", "large"], ["jit", "not_jit"])
    def test_t5_model(self, configuration, name) -> None:
        expected_asset_name = f"t5.{configuration}.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        is_jit = name == "jit"
        if configuration == "base":
            t5_model = T5_BASE
        elif configuration == "small":
            t5_model = T5_SMALL
        elif configuration == "large":
            t5_model = T5_LARGE

        self._t5_model(is_jit=is_jit, t5_model=t5_model, expected_asset_name=expected_asset_name, test_text=test_text)

    @nested_params(["base", "small", "large"], ["jit", "not_jit"])
    def test_t5_generation_model(self, configuration, name) -> None:
        expected_asset_name = f"t5.{configuration}.generation.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        is_jit = name == "jit"
        if configuration == "base":
            t5_model = T5_BASE_GENERATION
        elif configuration == "small":
            t5_model = T5_SMALL_GENERATION
        elif configuration == "large":
            t5_model = T5_LARGE_GENERATION

        self._t5_model(is_jit=is_jit, t5_model=t5_model, expected_asset_name=expected_asset_name, test_text=test_text)
