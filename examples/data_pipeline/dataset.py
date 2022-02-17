import torch
from torchtext.datasets import DATASETS


class BatchTextClassificationData(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, batch_size=16):
        super(BatchTextClassificationData, self).__init__()
        self._iterator = DATASETS[dataset_name](split="train")
        self.batch_size = batch_size

    def __iter__(self):
        _data = []
        for i, item in enumerate(self._iterator):
            _data.append(item)
            if len(_data) >= self.batch_size:
                yield _data
                _data = []
        if len(_data) > 0:
            yield _data
