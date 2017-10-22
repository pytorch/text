import os

from .. import data


class SequenceLabelingDataset(data.Dataset):
    """Defines a dataset for sequence labeling. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_labeling.py on how to use this class.
    """

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
    def load_default_dataset(cls, fields, root=".data"):
        """Downloads and loads the Universal Dependencies Version 2 POS Tagged
        data.
        """

        path = cls.download(root)  # .data/sequence-tagging/en-ud-v2
        return cls.splits(fields=fields, path=path,
                          train="en-ud-tag.v2.train.txt",
                          validation="en-ud-tag.v2.dev.txt",
                          test="en-ud-tag.v2.test.txt")
