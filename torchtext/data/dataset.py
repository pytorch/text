import torch.utils.data
import warnings

class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, fields, filter_pred=None):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning


class TabularDataset(Dataset):
    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning
