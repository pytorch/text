import torch
from torchtext.experimental import datasets
import torch.utils._benchmark as benchmark_utils

def benchmark_dataset(name, Dataset):

    print(name)
    print(Dataset)
    Dataset()
    timer = benchmark_utils.Timer(
        stmt="D()",
        globals={"D": Dataset},
        label="Benchmarking dataset %s".format(name),
    )
    timer.blocked_autorange()

    # for i in range(3):
    #     print(f"Run: {i}\n{'-' * 40}")
    #     print(f"timeit:\n{timer.timeit(10000)}\n")
    #     print(f"autorange:\n{timer.blocked_autorange()}\n\n")


if __name__ == "__main__":
    # for dataset in dir(datasets)
    for name, Dataset in datasets.DATASETS.items():
        benchmark_dataset(name, Dataset)
