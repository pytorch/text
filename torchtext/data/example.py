import csv
import json


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def fromJSON(cls, data, fields):
        return cls.fromdict(json.loads(data), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key in data and vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

    @classmethod
    def fromTSV(cls, data, fields):
        if data[-1] == '\n':
            data = data[:-1]
        return cls.fromlist(data.split('\t'), fields)

    @classmethod
    def fromCSV(cls, data, fields):
        if data[-1] == '\n':
            data = data[:-1]
        return cls.fromlist(list(csv.reader([data]))[0], fields)

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        try:
            from nltk.tree import Tree
        except ImportError:
            print('''Please install NLTK:
    $ pip install nltk''')
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [cls.fromlist(
                [t.leaves(), t.label()], fields) for t in tree.subtrees()]
        return cls.fromlist([tree.leaves(), tree.label()], fields)
