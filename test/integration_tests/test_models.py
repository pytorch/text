import pytest  # noqa: F401
import torch
from parameterized import parameterized, parameterized_class
from torchtext.models import (
    ROBERTA_BASE_ENCODER,
    ROBERTA_LARGE_ENCODER,
    ROBERTA_DISTILLED_ENCODER,
    XLMR_BASE_ENCODER,
    XLMR_LARGE_ENCODER,
)
from torchtext_unittest.common.assets import get_asset_path
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase

BUNDLERS = {
    "xlmr_base": XLMR_BASE_ENCODER,
    "xlmr_large": XLMR_LARGE_ENCODER,
    "roberta_base": ROBERTA_BASE_ENCODER,
    "roberta_large": ROBERTA_LARGE_ENCODER,
    "roberta_distilled": ROBERTA_DISTILLED_ENCODER,
}


@parameterized_class(
    ("model_name",),
    [
        ("xlmr_base",),
        ("xlmr_large",),
        ("roberta_base",),
        ("roberta_large",),
        ("roberta_distilled",),
    ],
)
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

    @parameterized.expand(["jit", "not_jit"])
    def test_models(self, name):
        configuration, type = self.model_name.split("_")

        expected_asset_name = f"{configuration}.{type}.output.pt"
        is_jit = name == "jit"
        if configuration == "xlmr":
            test_text = "XLMR base Model Comparison"
        else:
            test_text = "Roberta base Model Comparison"

        self._roberta_encoders(
            is_jit=is_jit,
            encoder=BUNDLERS[configuration + "_" + type],
            expected_asset_name=expected_asset_name,
            test_text=test_text,
        )
