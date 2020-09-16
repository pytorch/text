import warnings


class Example(object):
    @classmethod
    def fromJSON(cls, data, fields):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning

    @classmethod
    def fromdict(cls, data, fields):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning

    @classmethod
    def fromCSV(cls, data, fields, field_to_index=None):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning

    @classmethod
    def fromlist(cls, data, fields):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        warnings.warn('{} class has been retired and moved to torchtext.legacy. Please import from torchtext.legacy.data if you still want it.'.format(self.__class__.__name__), UserWarning)
        raise ImportWarning
