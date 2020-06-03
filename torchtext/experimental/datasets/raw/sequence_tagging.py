import torch

from torchtext.utils import download_from_url, extract_archive

URLS = {
    "UDPOS":
    'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip',
    "CoNLL2000Chunking": [
        'https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz',
        'https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz'
    ]
}


def _create_data_from_iob(data_path, separator="\t"):
    with open(data_path, encoding="utf-8") as input_file:
        columns = []
        for line in input_file:
            line = line.strip()
            if line == "":
                if columns:
                    yield columns
                columns = []
            else:
                for i, column in enumerate(line.split(separator)):
                    if len(columns) < i + 1:
                        columns.append([])
                    columns[i].append(column)


def _setup_datasets(dataset_name,
                    train_filename,
                    valid_filename,
                    test_filename,
                    separator,
                    root=".data"):

    extracted_files = []
    if isinstance(URLS[dataset_name], list):
        for f in URLS[dataset_name]:
            dataset_tar = download_from_url(f, root=root)
            extracted_files.extend(extract_archive(dataset_tar))
    elif isinstance(URLS[dataset_name], str):
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files.extend(extract_archive(dataset_tar))
    else:
        raise ValueError(
            "URLS for {} has to be in a form or list or string".format(
                dataset_name))

    data_filenames = dict()
    for fname in extracted_files:
        if train_filename and train_filename in fname:
            data_filenames["train"] = fname
        else:
            data_filenames["train"] = None

        if valid_filename and valid_filename in fname:
            data_filenames["valid"] = fname
        else:
            data_filenames["valid"] = None

        if test_filename and test_filename in fname:
            data_filenames["test"] = fname
        else:
            data_filenames["test"] = None

    datasets = []
    for key in data_filenames.keys():
        if data_filenames[key] is not None:
            datasets.append(
                RawSequenceTaggingIterableDataset(
                    _create_data_from_iob(data_filenames[key], separator)))

    return datasets


class RawSequenceTaggingIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text sequence tagging iterable datasets.
    """
    def __init__(self, iterator):
        super(RawSequenceTaggingIterableDataset).__init__()

        self._iterator = iterator
        self.has_setup = False
        self.start = 0
        self.num_lines = None

    def setup_iter(self, start=0, num_lines=None):
        self.start = start
        self.num_lines = num_lines
        self.has_setup = True

    def __iter__(self):
        if not self.has_setup:
            self.setup_iter()

        for i, item in enumerate(self._iterator):
            if i >= self.start:
                yield item
            if (self.num_lines is not None) and (i == (self.start +
                                                       self.num_lines)):
                break

    def get_iterator(self):
        return self._iterator


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
                      valid_filename=None,
                      test_filename="test.txt",
                      root=".data"):
    """ CoNLL 2000 Chunking Dataset
    """
    return _setup_datasets(dataset_name="CoNLL2000Chunking",
                           root=root,
                           train_filename=train_filename,
                           valid_filename=valid_filename,
                           test_filename=test_filename,
                           separator=' ')


DATASETS = {"UDPOS": UDPOS, "CoNLL2000Chunking": CoNLL2000Chunking}
