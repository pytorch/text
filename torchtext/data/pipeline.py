class Pipeline(object):
    def __init__(self, convert_token=None):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning
