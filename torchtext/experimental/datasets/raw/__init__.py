import importlib
from .wmtnewscrawl import WMTNewsCrawl
from .multi30k import Multi30k
from .wmt14 import WMT14

DATASETS = {
    'WMTNewsCrawl': WMTNewsCrawl,
    'Multi30k': Multi30k,
    'WMT14': WMT14,
}

URLS = {}
NUM_LINES = {}
MD5 = {}
for dataset in DATASETS:
    dataset_module_path = "torchtext.experimental.datasets.raw." + dataset.lower()
    dataset_module = importlib.import_module(dataset_module_path)
    URLS[dataset] = dataset_module.URL
    NUM_LINES[dataset] = dataset_module.NUM_LINES
    MD5[dataset] = dataset_module.MD5

__all__ = sorted(list(map(str, DATASETS.keys())))
