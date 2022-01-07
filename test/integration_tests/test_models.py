import torch
import torchtext

from ..common.assets import get_asset_path
from ..common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):
    def test_xlmr_base(self):
        asset_path = get_asset_path("xlmr.base.output.pt")
        test_text = "XLMR base Model Comparison"

        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        transform = xlmr_base.transform()
        model = xlmr_base.get_model()
        model = model.eval()

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_base_jit(self):
        asset_path = get_asset_path("xlmr.base.output.pt")
        test_text = "XLMR base Model Comparison"

        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        transform = xlmr_base.transform()
        transform_jit = torch.jit.script(transform)
        model = xlmr_base.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)

        model_input = torch.tensor(transform_jit([test_text]))
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_large(self):
        asset_path = get_asset_path("xlmr.large.output.pt")
        test_text = "XLMR base Model Comparison"

        xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
        transform = xlmr_large.transform()
        model = xlmr_large.get_model()
        model = model.eval()

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_large_jit(self):
        asset_path = get_asset_path("xlmr.large.output.pt")
        test_text = "XLMR base Model Comparison"

        xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
        transform = xlmr_large.transform()
        transform_jit = torch.jit.script(transform)
        model = xlmr_large.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)

        model_input = torch.tensor(transform_jit([test_text]))
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)
