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

    def __init__(self, name, full_num_lines, iterator, offset=0, stride=1):
        """Initiate text-classification dataset.
        """
        super(RawTextIterableDataset, self).__init__()
        self.name = name
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.has_setup = False
        self.start = offset
        self.num_lines = None
        self.setup_iter(start=offset, num_lines=full_num_lines - offset)
        self.stride = stride

    def setup_iter(self, start=0, num_lines=None):
        self.start = start
        self.num_lines = num_lines
        if num_lines and self.start + self.num_lines > self.full_num_lines:
            raise ValueError("Requested start {} and num_lines {} exceeds available number of lines {}".format(
                self.start, self.num_lines, self.full_num_lines))
        self.has_setup = True

    def __iter__(self):
        if not self.has_setup:
            self.setup_iter()

        for i, item in enumerate(self._iterator):
            if i < self.start:
                continue
            if (i - self.start) % self.stride != 0:
                continue
            if self.num_lines and i >= (self.start + self.num_lines):
                break
            yield item

    def __len__(self):
        if self.has_setup:
            return self.num_lines
        return self.full_num_lines

    def get_iterator(self):
        return self._iterator
