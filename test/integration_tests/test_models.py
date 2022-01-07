import torch
import torchtext

from ..common.assets import get_asset_path
from ..common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):
    def test_roberta_base(self):
        asset_path = get_asset_path("roberta.base.output.pt")
        test_text = "Roberta base Model Comparison"

        roberta_base = torchtext.models.ROBERTA_BASE_ENCODER
        transform = roberta_base.transform()
        model = roberta_base.get_model()
        model = model.eval()

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_roberta_base_jit(self):
        asset_path = get_asset_path("roberta.base.output.pt")
        test_text = "Roberta base Model Comparison"

        roberta_base = torchtext.models.ROBERTA_BASE_ENCODER
        transform = roberta_base.transform()
        transform_jit = torch.jit.script(transform)
        model = roberta_base.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)

        model_input = torch.tensor(transform_jit([test_text]))
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_roberta_large(self):
        asset_path = get_asset_path("roberta.large.output.pt")
        test_text = "Roberta base Model Comparison"

        roberta_large = torchtext.models.ROBERTA_LARGE_ENCODER
        transform = roberta_large.transform()
        model = roberta_large.get_model()
        model = model.eval()

        model_input = torch.tensor(transform([test_text]))
        actual = model(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)

    def test_roberta_large_jit(self):
        asset_path = get_asset_path("roberta.large.output.pt")
        test_text = "Roberta base Model Comparison"

        roberta_large = torchtext.models.ROBERTA_LARGE_ENCODER
        transform = roberta_large.transform()
        transform_jit = torch.jit.script(transform)
        model = roberta_large.get_model()
        model = model.eval()
        model_jit = torch.jit.script(model)

        model_input = torch.tensor(transform_jit([test_text]))
        actual = model_jit(model_input)
        expected = torch.load(asset_path)
        torch.testing.assert_close(actual, expected)
