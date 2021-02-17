import warnings


class Batch(object):
    def __init__(self, data=None, dataset=None, device=None):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning
