import torch

from torchtext.experimental.datasets import raw
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.functional import (
    vocab_func,
    totensor,
    sequential_transforms,
)


def _build_vocab(data):
    total_columns = len(data[0])
    data_list = [[] for _ in range(total_columns)]
    vocabs = []

    for line in data:
        for idx, col in enumerate(line):
            data_list[idx].append(col)

    for it in data_list:
        vocabs.append(build_vocab_from_iterator(it))

    return vocabs


def _setup_datasets(dataset_name,
                    root=".data",
                    vocabs=None,
                    data_select=("train", "valid", "test")):
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(("train", "test"))):
        raise TypeError("Given data selection {} is not supported!".format(data_select))

    train, val, test = DATASETS[dataset_name](root=root)
    raw_data = {
        "train": [line for line in train] if train else None,
        "valid": [line for line in val] if val else None,
        "test": [line for line in test] if test else None
    }

    if vocabs is None:
        if "train" not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocabs = _build_vocab(raw_data["train"])
    else:
        if not isinstance(vocabs, list):
            raise TypeError("vocabs must be an instance of list")

        # Find data that's not None
        notnone_data = None
        for key in raw_data.keys():
            if raw_data[key] is not None:
                notnone_data = raw_data[key]
                break
        if len(vocabs) != len(notnone_data[0]):
            raise ValueError(
                "Number of vocabs must match the number of columns "
                "in the data")

    transformers = [
        sequential_transforms(vocab_func(vocabs[idx]),
                              totensor(dtype=torch.long))
        for idx in range(len(vocabs))
    ]

    datasets = []
    for item in data_select:
        if raw_data[item] is not None:
            datasets.append(
                SequenceTaggingDataset(raw_data[item], vocabs, transformers))

    return datasets


class SequenceTaggingDataset(torch.utils.data.Dataset):
    """Defines an abstraction for raw text sequence tagging iterable datasets.
    Currently, we only support the following datasets:
        - UDPOS
        - CoNLL2000Chunking
    """
    def __init__(self, data, vocabs, transforms):
        """Initiate sequence tagging dataset.
        Arguments:
            data: a list of word and its respective tags. Example:
                [[word, POS, dep_parsing label, ...]]
            vocabs: a list of vocabularies for its respective tags.
                The number of vocabs must be the same as the number of columns
                found in the data.
            transforms: a list of string transforms for words and tags.
                The number of transforms must be the same as the number of columns
                    found in the data.
        """

        super(SequenceTaggingDataset, self).__init__()
        self.data = data
        self.vocabs = vocabs
        self.transforms = transforms

        if len(self.data[0]) != len(self.vocabs):
            raise ValueError("vocabs must hahve the same number of columns "
                             "as the data")

        if len(self.data[0]) != len(self.transforms):
            raise ValueError("vocabs must hahve the same number of columns "
                             "as the data")

    def __getitem__(self, i):
        curr_data = self.data[i]
        if len(curr_data) != len(self.transforms):
            raise ValueError("data must have the same number of columns "
                             "with transforms function")
        return [self.transforms[idx](curr_data[idx]) for idx in range(self.transforms)]

    def __len__(self):
        return len(self.data)

    def get_vocabs(self):
        return self.vocabs


def UDPOS(*args, **kwargs):
    """ Universal Dependencies English Web Treebank

    Separately returns the training, validation, and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        vocabs: A list of voabularies for each columns in the dataset. Must be in an
            instance of List
            Default: None
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'valid', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.datasets.raw import UDPOS
        >>> train_dataset, valid_dataset, test_dataset = UDPOS()
    """
    return _setup_datasets(*(("UDPOS", ) + args), **kwargs)


def CoNLL2000Chunking(*args, **kwargs):
    """ CoNLL 2000 Chunking Dataset

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        vocabs: A list of voabularies for each columns in the dataset. Must be in an
            instance of List
            Default: None
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'valid', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.datasets.raw import CoNLL2000Chunking
        >>> train_dataset, valid_dataset, test_dataset = CoNLL2000Chunking()
    """
    return _setup_datasets(*(("CoNLL2000Chunking", ) + args), **kwargs)


DATASETS = {"UDPOS": raw.UDPOS, "CoNLL2000Chunking": raw.CoNLL2000Chunking}
