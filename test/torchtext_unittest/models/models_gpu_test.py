import torch

from ..common.case_utils import skipIfNoCuda
from ..common.torchtext_test_case import TorchtextTestCase
from .models_test_impl import BaseTestModels


@skipIfNoCuda
class TestModels32GPUTest(BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("cuda")
