import codecs
from nltk.tree import Tree
import os

from .. import data

class TreeDataset(data.Dataset):
    """Defines a Dataset of trees stores in files seperated by newline"""

    def __init__(self, path, fields, fine=False, neutral=True,
                 minimum_sentence_length=4, **kwargs):
        """Create a TreeDataset given a path, field list and dataset specific
               arguments on how to preprocess the dataset. Notice that when
               using the train.txt set, subtrees will be generated as training
               examples.
        Arguments:
            path: Path to the data file.
            fields: dictionary whose keys are 'sentence1' for the sentence and
                'label1' for the label.
            fine: (Default: False) if True use all classes, if False subject
                classes 0, 1 -> <Negative> and 3, 4 -> <Positive>.
            neutral: (Default: True) This is only used if fine is True. Given
                fine is True, if True, do nothing. if False, do not include
                samples with class == 2.
            minimum_sentence_length: (Default: 4) remove all sentence below
                the integer threshold. Set to None for skipping this criteria.
        """
        self.fine = fine
        self.neutral = neutral
        self.minimum_sentence_length = minimum_sentence_length
        samples = self.read_file(path) # list of samples
        # One sample equals {'sentence1': "A large boat ..",
        #                    'label1': '3'}
        examples = [data.Example.fromdict(sample, fields) for sample
                    in samples]
        fields = fields.values()
        super(TreeDataset, self).__init__(examples, fields, **kwargs)

    def read_file(self, path):
        """Reads a file given a path, checks if base_name of the path is in
           ['train', 'dev', 'test'] and start loading and formatting the tree
           with nltk. If base_name is 'train', then load subtrees as well.
           Feed the tree into self.read_tree, which returns a sample, and
           return all samples in the file."""
        # do we want similar docs as to above in all functions?
        # should read_file, and all methods below, be a classmethod?
        base_path = os.path.basename(path)  # train.txt/valid.txt/test.txt
        split_name = os.path.splitext(base_path)[0]  # ['train', '.txt'][0]
        splits_allowed = ['train', 'dev', 'test']
        if split_name not in splits_allowed:
            print("split is %s and not in," % split_name, splits_allowed)
            raise
        if split_name == 'train':
            subtrees = True
        else:
            subtrees = False
        samples = []
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                tree = Tree.fromstring(line)
                if subtrees:
                    for subtree in tree.subtrees():
                        sample = self.read_tree(subtree, subtrees)
                        if sample is not None:
                            samples.append(sample)
                else:
                    sample = self.read_tree(tree)
                    if sample is not None:
                        samples.append(sample)
        return samples

    def read_tree(self, tree, subtrees=False):
        if self.fine:
            label = self.fine_class_func(tree.label())
        else:
            label = self.binary_class_func(tree.label())
        if label is not None:
            sentence = " ".join(tree.leaves())
            sample_dict = self.convert_to_dict((label, sentence))
            if self.minimum_sentence_length is not None:
                if (len(sample_dict['sentence1']) <
                    self.minimum_sentence_length):
                    return None
            return sample_dict
        else:
            return None

    def binary_class_func(self, y):
        # this binary function is hardcoded for sstb
        if y in ("0", "1"):
            return "<negative>"
        elif y in ("3", "4"):
            return "<positive>"
        else:
            return None

    def fine_class_func(self, y):
        # hardcoded for sstb
        if (not self.neutral) and (y in ('2')):
            return None
        else:
            return y

    def convert_to_dict(self, example):
        # hardcoded for sentence, label pair.
        label, sentence = example
        qa = {}
        qa['label1'] = label # label1 as it has to match Example.todict()
        qa['sentence1'] = sentence
        return qa


class SSTBDataset(data.ZipDataset, TreeDataset):

    url = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    filename = 'trainDevTestTrees_PTB.zip'
    dirname = 'trees/'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.sentence), 1)# perhaps interleave keys is not required here?

    @classmethod
    def splits(cls, text_field, label_field, root='.', train='train.txt',
               validation='dev.txt', test='test.txt', **kwargs):
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
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        return super(SSTBDataset, cls).splits(
            path, train, validation, test,
            fields={'sentence1': ('sentence', text_field),
                    'label1': ('label', label_field)}, **kwargs)

    @classmethod
    def iter(cls, batch_size=32, device=0, root='.', wv_path=None, **kwargs):
        """Creater iterator objects for splits of the SSTB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            wv_path: The path to the word vector file that will be loaded into
                the vector attribute of the created vocabulary (accessible
                as train.dataset.fields['sentence1'].vocab.vectors).
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train) # fails if vw_path is None
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
