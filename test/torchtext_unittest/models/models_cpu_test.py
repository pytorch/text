import torch

from ..common.torchtext_test_case import TorchtextTestCase
from .models_test_impl import BaseTestModels


class TestModels32CPUTest(BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
