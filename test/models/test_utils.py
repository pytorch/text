import torch
from torchtext.experimental.models.utils import count_model_param
from ..common.torchtext_test_case import TorchtextTestCase


class TestModelsUtils(TorchtextTestCase):
    def test_count_model_parameters_func(self):
        model = torch.nn.Embedding(100, 200)
        self.assertEqual(count_model_param(model, unit=10**3), 20.0)
