import torch

from torchtext.experimental.datasets import raw
from torchtext.experimental.functional import (
    vocab_func,
    totensor,
    sequential_transforms,
)

def _build_vocab(data):
    
    for line in data:
        for col in line:



def _setup_datasets(
        dataset_name,
        train_filename,
        valid_filename,
        test_filename,
        separator,
        data_select=("train", "valid", "test"),
        root=".data",
        vocab=(None, None),
        tokenizer=None,
):

    text_transform = []
    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")
    train, val, test = DATASETS[dataset_name](train_filename=train_filename,
                                              valid_filename=valid_filename,
                                              test_filename=test_filename,
                                              root=root)
    raw_data = {
        "train": [line for line in train],
        "valid": [line for line in val] if val else None,
        "test": [line for line in test] if test else None
    }

    text_transform = sequential_transforms(tokenizer)


    data_filenames = dict()
    for fname in extracted_files:
        if train_filename and train_filename in fname:
            data_filenames["train"] = fname
        if valid_filename and valid_filename in fname:
            data_filenames["valid"] = fname
        if test_filename and test_filename in fname:
            data_filenames["test"] = fname

    datasets = []
    for key in data_filenames.keys():
        if data_filenames[key] is not None:
            datasets.append(
                RawSequenceTaggingIterableDataset(
                    _create_data_from_iob(data_filenames[key], separator)))

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
          root=".data"):
    """ Universal Dependencies English Web Treebank.
    """
    return _setup_datasets(dataset_name="UDPOS",
                           root=root,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           separator="\t")


def CoNLL2000Chunking(train_filename="train.txt",
                      test_filename="test.txt",
                      root=".data"):
    """ CoNLL 2000 Chunking Dataset
    """
    return _setup_datasets(dataset_name="CoNLL2000Chunking",
                           root=root,
                           train_filename=train_filename,
                           valid_filename=None,
                           test_filename=test_filename,
                           separator=' ')


DATASETS = {"UDPOS": raw.UDPOS, "CoNLL2000Chunking": raw.CoNLL2000Chunking}
