class Pipeline(object):
    def __init__(self, convert_token=None):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning
