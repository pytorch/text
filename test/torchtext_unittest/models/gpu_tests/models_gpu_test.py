import unittest

import pytest
import torch
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase
from torchtext_unittest.models.models_test_impl import BaseTestModels


@pytest.mark.gpu_test
@unittest.skipIf(not torch.cuda.is_available(), reason="CUDA is not available")
class TestModels32GPU(BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
