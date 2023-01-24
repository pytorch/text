import pytest
import torch

from ..common.torchtext_test_case import TorchtextTestCase
from .models_test_impl import BaseTestModels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestModels32GPU(BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
