import torch
from torchtext.models import (
    XLMR_BASE_ENCODER,
    XLMR_LARGE_ENCODER,
    ROBERTA_BASE_ENCODER,
    ROBERTA_LARGE_ENCODER,
)

from ..common.assets import get_asset_path
from ..common.parameterized_utils import nested_params
from ..common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):
    @nested_params(
        [
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
        ],
        [True, False],
    )
    def test_model(self, model_args, is_jit):
        expected_asset_name, test_text, model_bundler = model_args

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
