import torch

from ..common.torchtext_test_case import TorchtextTestCase
from .roberta_models_test_impl import RobertaBaseTestModels
from .t5_models_test_impl import T5BaseTestModels


class TestModels32CPU(RobertaBaseTestModels, T5BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("cpu")
