from torchtext._internal.module_utils import is_module_available


def check_for_torchdata():
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install torchdata <= 0.9.0: https://github.com/pytorch/data"
        )


__all__ = [
    "check_for_torchdata",
    "dataloader2",
    "datapipes",
]
