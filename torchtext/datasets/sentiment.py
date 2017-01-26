from nltk.tree import Tree
import os

from .. import data


class SSTDataset(data.ZipDataset):

    url = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    filename = 'trainDevTestTrees_PTB.zip'
    dirname = 'trees'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, subtrees=False,
                 fine_grained=False, **kwargs):
        """Create an SSTDataset given a path and fields.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]
        label_field.preprocessing = data.Pipeline(get_label_str)
        with open(os.path.expanduser(path)) as f:
            if subtrees:
                examples = [ex for line in f for ex in
                            data.Example.fromtree(line, fields, True)]
            else:
                examples = [data.Example.fromtree(line, fields) for line in f]
        super(SSTDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.',
               train='train.txt', validation='dev.txt', test='test.txt',
               train_subtrees=False, **kwargs):
        """Create dataset objects for splits of the SSTB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.txt'.
            train_subtrees: Whether to use all subtrees in the training set.
                Default: False.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)

        train_data = None if train is None else cls(
            path + train, text_field, label_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            path + validation, text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            path + test, text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_path=None, **kwargs):
        """Creater iterator objects for splits of the SSTB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            wv_path: The path to the word vector file that will be loaded into
                the vector attribute of the created vocabulary(accessible
                as train.dataset.fields['sentence1'].vocab.vectors).
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
