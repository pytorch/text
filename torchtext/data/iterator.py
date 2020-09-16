import logging
import warnings

logger = logging.getLogger(__name__)


class Iterator(object):
    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning


class BPTTIterator(Iterator):
    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning


class BucketIterator(Iterator):
    def create_batches(self):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning
