import torch


class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, name, num_lines, iterator, lazy=False):
        """Initiate text-classification dataset.
        """
        super(RawTextIterableDataset, self).__init__()
        self.name = name
        self.iterator = iterator
        self.has_setup = False
        self.start = 0
        self.num_lines = num_lines
        self.lazy = lazy

    def setup_iter(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            if not self.lazy:
                raise RuntimeError("Given iterator doesn't support DataLoader.")
            chunk = int(self.num_lines / worker_info.num_workers)
            self.start = chunk * worker_info.id
            self.num_lines = chunk
            if worker_info.id == worker_info.num_workers - 1:
                # The last worker needs to pick up some extra lines
                # if the number of lines aren't exactly divisible
                # by the number of workers.
                # Each epoch we loose an 'extra' number of lines.
                extra = self.num_lines % worker_info.num_workers
                self.num_lines += extra
        else:
            self.start = 0

        if self.lazy:
            self.iterator = self.iterator()

        self.has_setup = True

    def __iter__(self):
        if not self.has_setup:
            self.setup_iter()

        for i, item in enumerate(self.iterator):
            if i < self.start:
                continue
            if i > (self.start + self.num_lines):
                break
            yield item

    def __len__(self):
        return self.num_lines

    def get_iterator(self):
        return self.iterator
