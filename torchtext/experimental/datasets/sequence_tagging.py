import torch

from torchtext.experimental.datasets import raw
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.functional import (
    vocab_func,
    totensor,
    sequential_transforms,
)


def _build_vocab(data, word_transform):
    total_columns = len(data[0])
    data_list = [[] for _ in range(total_columns)]
    vocabs = []

    for line in data:
        for idx, col in enumerate(line):
            if idx == 0:
                col = word_transform(col) if word_transform else col
                data_list[idx].append(col)
            else:
                data_list[idx].append(col)

    for it in data_list:
        vocabs.append(build_vocab_from_iterator(it))

    return vocabs


def _setup_datasets(dataset_name,
                    train_filename,
                    valid_filename,
                    test_filename,
                    separator,
                    data_select=("train", "valid", "test"),
                    root=".data",
                    vocabs=None,
                    word_tokenizer=None):
    train, val, test = DATASETS[dataset_name](train_filename=train_filename,
                                              valid_filename=valid_filename,
                                              test_filename=test_filename,
                                              root=root)
    raw_data = {
        "train": [line for line in train] if train else None,
        "valid": [line for line in val] if val else None,
        "test": [line for line in test] if test else None
    }

    word_transform = None
    if word_tokenizer:
        word_transform = sequential_transforms(word_tokenizer)

    if vocabs is None:
        if "train" not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocabs = _build_vocab(raw_data["train"], word_transform)
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

    if word_transform:
        word_transform = sequential_transforms(word_transform,
                                               vocab_func(vocabs[0]),
                                               totensor(dtype=torch.long))
    else:
        word_transform = sequential_transforms(vocab_func(vocabs[0]),
                                               totensor(dtype=torch.long))
    labels_transforms = [
        sequential_transforms(vocab_func(vocabs[idx + 1]),
                              totensor(dtype=torch.long))
        for idx in range(len(vocabs) - 1)
    ]
    transformers = [word_transform, *labels_transforms]

    datasets = []
    for item in data_select:
        if raw_data[item] is not None:
            datasets.append(
                SequenceTaggingDataset(raw_data[item], vocabs, transformers))

    return datasets


class SequenceTaggingDataset(torch.utils.data.Dataset):
    """Defines an abstraction for raw text sequence tagging iterable datasets.
    """
    def __init__(self, data, vocab, transforms):
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


def UDPOS(train_filename="en-ud-tag.v2.train.txt",
          valid_filename="en-ud-tag.v2.dev.txt",
          test_filename="en-ud-tag.v2.test.txt",
          data_select=("train", "valid", "test"),
          root=".data",
          vocabs=None,
          word_tokenizer=None):
    """ Universal Dependencies English Web Treebank.
    """
    return _setup_datasets(dataset_name="UDPOS",
                           root=root,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           separator="\t",
                           data_select=data_select,
                           vocabs=vocabs,
                           word_tokenizer=word_tokenizer)


def CoNLL2000Chunking(train_filename="train.txt",
                      test_filename="test.txt",
                      data_select=("train", "valid", "test"),
                      root=".data",
                      vocabs=None,
                      word_tokenizer=None):
    """ CoNLL 2000 Chunking Dataset
    """
    return _setup_datasets(dataset_name="CoNLL2000Chunking",
                           root=root,
                           train_filename=train_filename,
                           valid_filename=None,
                           test_filename=test_filename,
                           separator=' ',
                           data_select=data_select,
                           vocabs=vocabs,
                           word_tokenizer=word_tokenizer)


DATASETS = {"UDPOS": raw.UDPOS, "CoNLL2000Chunking": raw.CoNLL2000Chunking}
