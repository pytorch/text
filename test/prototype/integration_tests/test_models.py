import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.prototype.models import (
    T5_BASE_ENCODER,
    T5_BASE,
)


class TestT5(TorchtextTestCase):
    def _t5_model(self, t5_model, expected_asset_name, encoder_input):
        """Verify that pre-trained T5 models in torchtext produce
        the same output as the HuggingFace reference implementation.
        """
        expected_asset_path = get_asset_path(expected_asset_name)
        model = t5_model.get_model()
        model = model.eval()

        if model.encoder_only:
            actual = model(encoder_input)["encoder_output"]
        else:
            actual = model(encoder_input)["decoder_output"]

        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected)

    def test_t5_base_encoder_model(self):
        expected_asset_name = "t5.base.encoder.output.pt"
        encoder_input = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]])
        self._t5_model(t5_model=T5_BASE_ENCODER, expected_asset_name=expected_asset_name, encoder_input=encoder_input)

    def test_t5_base_model(self):
        expected_asset_name = "t5.base.output.pt"
        encoder_input = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]])
        self._t5_model(t5_model=T5_BASE, expected_asset_name=expected_asset_name, encoder_input=encoder_input)
