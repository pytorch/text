import unittest

import pytest
import torch
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase
from torchtext_unittest.models.roberta_models_test_impl import RobertaBaseTestModels
from torchtext_unittest.models.t5_models_test_impl import T5BaseTestModels


@pytest.mark.gpu_test
@unittest.skipIf(not torch.cuda.is_available(), reason="CUDA is not available")
class TestModels32GPU(RobertaBaseTestModels, T5BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
