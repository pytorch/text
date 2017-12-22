from .. import data


class SequenceTaggingDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

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
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                      **kwargs)

class UDPOS(SequenceTaggingDataset):

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
    
    @classmethod
    def splits(cls, fields, root=".data", train="en-ud-tag.v2.train.txt",
               validation="en-ud-tag.v2.dev.txt",
               test="en-ud-tag.v2.test.txt", **kwargs):
        """Downloads and loads the Universal Dependencies Version 2 POS Tagged
        data.
        """

        return super(UDPOS, cls).splits(
            fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)
