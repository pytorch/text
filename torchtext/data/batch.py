from torch import typename
from torch.tensor import _TensorBase


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            self.fields = dataset.fields.keys()  # copy field names

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [x.__dict__[name] for x in data]
                    setattr(self, name, field.process(batch, device=device, train=train))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.__dict__:
            return 'Empty {} instance'.format(typename(self))

        var_strs = '\n'.join(['\t[.' + name + ']' + ":" + _short_str(getattr(self, name))
                              for name in self.fields])

        data_str = (' from {}'.format(self.dataset.name.upper())
                    if hasattr(self.dataset, 'name') else '')

        strt = '[{} of size {}{}]\n{}'.format(typename(self),
                                              self.batch_size, data_str, var_strs)
        return '\n' + strt


def _short_str(tensor):
    # unwrap variable to tensor
    if hasattr(tensor, 'data'):
        tensor = tensor.data

    # fallback in case of wrong argument type
    if issubclass(type(tensor), _TensorBase) is False:
        return str(tensor)

    # copied from torch _tensor_str
    size_str = 'x'.join(str(size) for size in tensor.size())
    device_str = '' if not tensor.is_cuda else \
        ' (GPU {})'.format(tensor.get_device())
    strt = '[{} of size {}{}]'.format(typename(tensor),
                                      size_str, device_str)
    return strt
