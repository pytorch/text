import torch
from torchtext.models import (
    ROBERTA_BASE_ENCODER,
    ROBERTA_LARGE_ENCODER,
    XLMR_BASE_ENCODER,
    XLMR_LARGE_ENCODER,
)

from ..common.assets import get_asset_path
from ..common.parameterized_utils import nested_params
from ..common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):
    def _xlmr_base_model(self, is_jit):
        """Verify pre-trained XLM-R and Roberta models in torchtext produce
        the same output as the reference implementation within fairseq
        """
        expected_asset_name, test_text, model_bundler = (
            "xlmr.base.output.pt",
            "XLMR base Model Comparison",
            XLMR_BASE_ENCODER,
        )

        expected_asset_path = get_asset_path(expected_asset_name)

        transform = model_bundler.transform()
        model = model_bundler.get_model()
        model = model.eval()

        if is_jit:
            transform = torch.jit.script(transform)
            model = torch.jit.script(model)

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_base_model(self):
        self._xlmr_base_model(is_jit=False)

    def test_xlmr_base_model_jit(self):
        self._xlmr_base_model(is_jit=True)

    def _xlmr_large_model(self, is_jit):
        """Verify pre-trained XLM-R and Roberta models in torchtext produce
        the same output as the reference implementation within fairseq
        """
        expected_asset_name, test_text, model_bundler = (
            "xlmr.large.output.pt",
            "XLMR base Model Comparison",
            XLMR_LARGE_ENCODER,
        )

        expected_asset_path = get_asset_path(expected_asset_name)

        transform = model_bundler.transform()
        model = model_bundler.get_model()
        model = model.eval()

        if is_jit:
            transform = torch.jit.script(transform)
            model = torch.jit.script(model)

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_large_model(self):
        self._xlmr_large_model(is_jit=False)

    def test_xlmr_large_model_jit(self):
        self._xlmr_large_model(is_jit=True)
