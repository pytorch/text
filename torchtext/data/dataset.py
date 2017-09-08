import os
import zipfile

import torch.utils.data
import six
from six.moves import urllib

from .example import Example


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key: The key to use for sorting examples from this dataset in
            order to batch together examples with similar lengths and minimize
            padding.
        examples: The list of Examples in the dataset.
        fields: A dictionary containing the name of each column together with
            its corresponding Field object. Two columns with the same Field
            object will share a vocabulary.
    """

    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of examples and fields.

        Arguments:
            examples: List of Examples.
            fields: List of tuples of (name, field).
            filter_pred: Use only examples for which filter_pred(ex) is True,
                or use all examples if None. Default: None.
        """
        if filter_pred is not None:
            examples = list(filter(filter_pred, examples))
        self.examples = examples

        self.fields = dict(fields)

    @classmethod
    def splits(cls, path, train=None, validation=None, test=None, **kwargs):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            path: Common prefix of the splits' file paths.
            train: Suffix to add to path for the train set, or None for no
                train set. Default: None.
            validation: Suffix to add to path for the validation set, or None
                for no validation set. Default: None.
            test: Suffix to add to path for the test set, or None for no test
                set. Default: None.
            Remaining keyword arguments: Passed to the constructor of the
                dataset class being used.
        """
        train_data = None if train is None else cls(path + train, **kwargs)
        val_data = None if validation is None else cls(path + validation,
                                                       **kwargs)
        test_data = None if test is None else cls(path + test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path: Path to the data file.
            format: One of "CSV", "TSV", or "JSON" (case-insensitive).
            fields: For CSV and TSV formats, list of tuples of (name, field).
                The list should be in the same order as the columns in the CSV
                or TSV file, while tuples of (name, None) represent columns
                that will be ignored. For JSON format, dictionary whose keys
                are the JSON keys and whose values are tuples of (name, field).
                This allows the user to rename columns from their JSON key
                names or select a subset of columns to load while ignoring
                others not present in this dictionary.
        """
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromTSV, 'csv': Example.fromCSV}[format.lower()]

        with open(os.path.expanduser(path)) as f:
            examples = [
                make_example(line.decode('utf-8') if six.PY2 else line, fields)
                for line in f]

        if make_example in (Example.fromdict, Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


class ZipDataset(Dataset):
    """Defines a Dataset loaded from a downloadable zip archive.

    Attributes:
        url: URL where the zip archive can be downloaded.
        filename: Filename of the downloaded zip archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            zpath = os.path.join(root, cls.filename)
            if not os.path.isfile(zpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, zpath)
            with zipfile.ZipFile(zpath, 'r') as zfile:
                print('extracting')
                zfile.extractall(root)
        return os.path.join(path, '')
