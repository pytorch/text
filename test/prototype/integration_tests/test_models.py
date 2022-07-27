import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.prototype.models import (
    T5_BASE_ENCODER,
    T5_BASE,
    T5_BASE_GENERATION,
)


class TestT5(TorchtextTestCase):
    def _t5_model(self, t5_model, expected_asset_name, test_text):
        """Verify that pre-trained T5 models in torchtext produce
        the same output as the HuggingFace reference implementation.
        """
        expected_asset_path = get_asset_path(expected_asset_name)
        transform = t5_model.transform()
        model = t5_model.get_model()
        model = model.eval()

        model_input = transform(test_text)
        if model.encoder_only:
            actual = model(model_input)["encoder_output"]
        else:
            actual = model(model_input)["decoder_output"]

        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected, atol=1e-04, rtol=2.5e-06)

    def test_t5_base_encoder_model(self):
        expected_asset_name = "t5.base.encoder.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        self._t5_model(t5_model=T5_BASE_ENCODER, expected_asset_name=expected_asset_name, test_text=test_text)

    def test_t5_base_model(self):
        expected_asset_name = "t5.base.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        self._t5_model(t5_model=T5_BASE, expected_asset_name=expected_asset_name, test_text=test_text)

    def test_t5_base_generation_model(self):
        expected_asset_name = "t5.base.generation.output.pt"
        test_text = ["Hello world", "Attention rocks!"]
        self._t5_model(t5_model=T5_BASE_GENERATION, expected_asset_name=expected_asset_name, test_text=test_text)
