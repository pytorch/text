import os

from .. import data


class SequenceLabelingDataset(data.Dataset):

    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = ['https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip']
    dirname = 'en-ud-v2'
    name = 'sequence-labeling'

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
             if not callable(getattr(example, attr)) and \
                     not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, **kwargs):
        examples = []
        columns = []

        with open(path) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split("\t")):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(SequenceLabelingDataset, self).__init__(examples, fields,
                                                      **kwargs)


    @classmethod
    def load_default_dataset(cls, fields, path=".data"):
        path = cls.download(path) #.data/sequence-tagging/en-ud-v2
        return cls.splits(fields, path, train="en-ud-tag.v2.train.txt",
                          validation="en-ud-tag.v2.dev.txt",
                          test="en-ud-tag.v2.test.txt")

    @classmethod
    def splits(cls, fields, path, train=None, validation=None, test=None,
               **kwargs):
        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)