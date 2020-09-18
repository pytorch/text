import torch
from torchtext.experimental import datasets
import torch.utils._benchmark as benchmark_utils


def benchmark_construction(name, Dataset):
    Dataset()
    timer = benchmark_utils.Timer(
        stmt="D()",
        globals={"D": Dataset},
        label="Benchmarking dataset %s".format(name),
    )
    timer.blocked_autorange()


if __name__ == "__main__":
    for name, Dataset in datasets.DATASETS.items():
        benchmark_construction(name, Dataset)
