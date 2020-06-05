import torch

from torchtext.experimental.datasets import raw
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.functional import (
    vocab_func,
    totensor,
    sequential_transforms,
)


def _build_vocab(data, text_transform):
    total_columns = len(data[0])
    data_list = [[] for _ in range(total_columns)]
    vocabs = []

    for line in data:
        for idx, col in enumerate(line):
            if idx == 0:
                col = text_transform(col) if text_transform else col
                data_list[idx].append(col)
            else:
                data_list[idx].append(col)

    for it in data_list:
        vocabs.append(build_vocab_from_iterator(it))

    return vocabs


def _setup_datasets(dataset_name,
                    root=".data",
                    vocabs=None,
                    tokenizer=None,
                    data_select=("train", "valid", "test")):
    train, val, test = DATASETS[dataset_name](root=root)
    raw_data = {
        "train": [line for line in train] if train else None,
        "valid": [line for line in val] if val else None,
        "test": [line for line in test] if test else None
    }

    text_transform = None
    if tokenizer:
        text_transform = sequential_transforms(tokenizer)

    if vocabs is None:
        if "train" not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocabs = _build_vocab(raw_data["train"], text_transform)
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

    if text_transform:
        text_transform = sequential_transforms(text_transform,
                                               vocab_func(vocabs[0]),
                                               totensor(dtype=torch.long))
    else:
        text_transform = sequential_transforms(vocab_func(vocabs[0]),
                                               totensor(dtype=torch.long))
    labels_transforms = [
        sequential_transforms(vocab_func(vocabs[idx + 1]),
                              totensor(dtype=torch.long))
        for idx in range(len(vocabs) - 1)
    ]
    transformers = [text_transform, *labels_transforms]

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
    def __init__(self, data, vocab, transforms):
        """Initiate sequence tagging dataset.
        Arguments:
            data: a list of word and its respective tags. Example:
                [[word, POS, dep_parsing label, ...]]
            vocab: Vocabulary object used for dataset.
            transforms: a list of string transforms for words and tags.
        """

        super(SequenceTaggingDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms

    def __getitem__(self, i):
        line = []
        for idx, transform in enumerate(self.transforms):
            line.append(transform(self.data[i][idx]))
        return line

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


def UDPOS(*args, **kwargs):
    """ Universal Dependencies English Web Treebank

    Separately returns the training, validation, and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        vocabs: A list of voabularies for each columns in the dataset. Must be in an
            instance of List
            Default: None
        tokenizer: The tokenizer used to preprocess word column in raw text data
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
        tokenizer: The tokenizer used to preprocess word column in raw text data
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
