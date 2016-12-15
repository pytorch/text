import os

from text.torchtext import data


def interleave_keys(a, b):
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)


class TranslationDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields):

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        self.examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    self.examples.append(
                        data.Example.fromlist([src_line, trg_line], fields))

        self.fields = dict(fields)

    @classmethod
    def splits(cls, path, train=None, dev=None, test=None, **kwargs):
        train_data = None if train is None else cls(path + train, **kwargs)
        dev_data = None if dev is None else cls(path + dev, **kwargs)
        test_data = None if test is None else cls(path + test, **kwargs)
        return tuple(d for d in (train_data, dev_data, test_data)
                     if d is not None)
