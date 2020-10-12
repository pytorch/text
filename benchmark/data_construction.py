from torchtext.experimental import datasets
import time


def benchmark_construction(name, Dataset):
    t0 = time.perf_counter()
    print(name, end='')
    d, = Dataset(data_select=('train',))
    print(" construction time {0:.2f}s".format(time.perf_counter() - t0))
    del d


def benchmark_raw_construction(name, Dataset):
    print(name, end='')
    if name in "WMTNewsCrawl":
        d = Dataset(data_select=('train',))
    else:
        d = Dataset()
    del d


if __name__ == "__main__":
    for name, Dataset in datasets.DATASETS.items():
        benchmark_construction(name, Dataset)
