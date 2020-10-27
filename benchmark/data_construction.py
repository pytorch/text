from torchtext.experimental import datasets
import time


def benchmark_construction(name, Dataset):
    t0 = time.perf_counter()
    print(name, end='')
    d, = Dataset(data_select=('train',))
    print(" construction time {0:.2f}s".format(time.perf_counter() - t0))
    del d


def count_iterable(dataset):
    iter_ = iter(dataset)
    return sum(1 for _ in iter_)


def benchmark_raw_construction(name, Dataset):
    print(name, end='')
    if name != "WMTNewsCrawl":
        d = Dataset()
        for item in d:
            print(item, count_iterable(item))
    del d


if __name__ == "__main__":
    for name, Dataset in datasets.raw.DATASETS.items():
        benchmark_raw_construction(name, Dataset)
