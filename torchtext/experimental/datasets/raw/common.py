import torch


def check_default_set(data_select, target_select):
    if isinstance(data_select, str):
        data_select = (data_select,)
    if not set(data_select).issubset(set(target_select)):
        raise TypeError('A subset of data selection {} is supported but {} is passed in'.format(target_select,
                                                                                                data_select))
    return data_select


class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, name, full_num_lines, iterator, offset=0):
        """Initiate text-classification dataset.
        """
        super(RawTextIterableDataset, self).__init__()
        self.name = name
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.start = offset
        self.num_lines = full_num_lines - offset

    def __iter__(self):
        for i, item in enumerate(self._iterator):
            if i < self.start:
                continue
            if self.num_lines and i >= (self.start + self.num_lines):
                break
            yield item

    def __next__(self):
        item = next(self._iterator)
        return item

    def __len__(self):
        return self.num_lines

    def get_iterator(self):
        return self._iterator
