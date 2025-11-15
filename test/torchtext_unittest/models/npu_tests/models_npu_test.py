import importlib
import unittest

import pytest
import torch
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase
from torchtext_unittest.models.roberta_models_test_impl import RobertaBaseTestModels
from torchtext_unittest.models.t5_models_test_impl import T5BaseTestModels


def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()
 

@pytest.mark.npu_test
@unittest.skipIf(not is_npu_available(), reason="Ascend NPU is not available")
class TestModels32NPU(RobertaBaseTestModels, T5BaseTestModels, TorchtextTestCase):
    dtype = torch.float32
    device = torch.device("npu")
