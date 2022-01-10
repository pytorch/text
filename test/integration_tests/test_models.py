import torch
from parameterized import parameterized
from torchtext.models import (
    XLMR_BASE_ENCODER,
    XLMR_LARGE_ENCODER,
    ROBERTA_BASE_ENCODER,
    ROBERTA_LARGE_ENCODER,
)

from ..common.assets import get_asset_path
from ..common.torchtext_test_case import TorchtextTestCase

TEST_MODELS_PARAMETERIZED_ARGS = [
    ("xlmr.base.output.pt", "XLMR base Model Comparison", XLMR_BASE_ENCODER),
    ("xlmr.large.output.pt", "XLMR base Model Comparison", XLMR_LARGE_ENCODER),
    (
        "roberta.base.output.pt",
        "Roberta base Model Comparison",
        ROBERTA_BASE_ENCODER,
    ),
    (
        "roberta.large.output.pt",
        "Roberta base Model Comparison",
        ROBERTA_LARGE_ENCODER,
    ),
]


class TestModels(TorchtextTestCase):
    @parameterized.expand(TEST_MODELS_PARAMETERIZED_ARGS)
    def test_model(self, expected_asset_name, test_text, model_bundler):
        expected_asset_path = get_asset_path(expected_asset_name)

        transform = model_bundler.transform()
        model = model_bundler.get_model()
        model = model.eval()

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected)

    @parameterized.expand(TEST_MODELS_PARAMETERIZED_ARGS)
    def test_model_jit(self, expected_asset_name, test_text, model_bundler):
        expected_asset_path = get_asset_path(expected_asset_name)

        transform = model_bundler.transform()
        transform_jit = torch.jit.script(transform)
        model = model_bundler.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)

        model_input = torch.tensor(transform_jit([test_text]))
        actual = model_jit(model_input)
        expected = torch.load(expected_asset_path)
        torch.testing.assert_close(actual, expected)
