import torchtext
import torch

from ..common.torchtext_test_case import TorchtextTestCase
from ..common.assets import get_asset_path


class TestModels(TorchtextTestCase):
    def test_xlmr_base_output(self):
        asset_name = "xlmr.base.output.pt"
        asset_path = get_asset_path(asset_name)
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_base.get_model()
        model = model.eval()
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_base_jit_output(self):
        asset_name = "xlmr.base.output.pt"
        asset_path = get_asset_path(asset_name)
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_base.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)
        model_input = torch.tensor([[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]])
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_xlmr_transform(self):
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        transform = xlmr_base.transform()
        test_text = "XLMR base Model Comparison"
        actual = transform([test_text])
        expected = [[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]]
        torch.testing.assert_close(actual, expected)

    def test_xlmr_transform_jit(self):
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        transform = xlmr_base.transform()
        transform_jit = torch.jit.script(transform)
        test_text = "XLMR base Model Comparison"
        actual = transform_jit([test_text])
        expected = [[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]]
        torch.testing.assert_close(actual, expected)
