import torch
from parameterized import parameterized
from torchtext.models import (
    ROBERTA_BASE_ENCODER,
    ROBERTA_LARGE_ENCODER,
    XLMR_BASE_ENCODER,
    XLMR_LARGE_ENCODER,
)

from ..common.assets import get_asset_path
from ..common.torchtext_test_case import TorchtextTestCase


class TestRobertaEncoders(TorchtextTestCase):
    def _roberta_encoders(self, is_jit, encoder, expected_asset_name, test_text):
        """Verify pre-trained XLM-R and Roberta models in torchtext produce
        the same output as the reference implementation within fairseq
        """
        expected_asset_path = get_asset_path(expected_asset_name)

        transform = encoder.transform()
        model = encoder.get_model()
        model = model.eval()

        if is_jit:
            transform = torch.jit.script(transform)
            model = torch.jit.script(model)

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected)

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_xlmr_base_model(self, name, is_jit):
        expected_asset_name = "xlmr.base.output.pt"
        test_text = "XLMR base Model Comparison"
        self._roberta_encoders(
            is_jit=is_jit,
            encoder=XLMR_BASE_ENCODER,
            expected_asset_name=expected_asset_name,
            test_text=test_text,
        )

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_xlmr_large_model(self, name, is_jit):
        expected_asset_name = "xlmr.large.output.pt"
        test_text = "XLMR base Model Comparison"
        self._roberta_encoders(
            is_jit=is_jit,
            encoder=XLMR_LARGE_ENCODER,
            expected_asset_name=expected_asset_name,
            test_text=test_text,
        )

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_roberta_base_model(self, name, is_jit):
        expected_asset_name = "roberta.base.output.pt"
        test_text = "Roberta base Model Comparison"
        self._roberta_encoders(
            is_jit=is_jit,
            encoder=ROBERTA_BASE_ENCODER,
            expected_asset_name=expected_asset_name,
            test_text=test_text,
        )

    @parameterized.expand([("jit", True), ("not_jit", False)])
    def test_robeta_large_model(self, name, is_jit):
        expected_asset_name = "roberta.large.output.pt"
        test_text = "Roberta base Model Comparison"
        self._roberta_encoders(
            is_jit=is_jit,
            encoder=ROBERTA_LARGE_ENCODER,
            expected_asset_name=expected_asset_name,
            test_text=test_text,
        )
