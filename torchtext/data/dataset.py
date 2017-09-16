import io
import os
import zipfile
import tarfile

import torch.utils.data
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
        """Create a dataset from a list of Examples and fields.

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

    @classmethod
    def download(cls, root):
        path = os.path.join(root, cls.name)
        if not os.path.isdir(path):
            for url in cls.urls:
                filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    urllib.request.urlretrieve(url, zpath)
                ext = os.path.splitext(filename)[-1]
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)
                elif ext in ['.gz', '.tgz']:
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
        return os.path.join(path, cls.dirname)


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
                names and also enables selecting a subset of columns to load
                (since JSON keys not present in the input dictionary are ignored).
        """
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromTSV, 'csv': Example.fromCSV}[format.lower()]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            examples = [make_example(line, fields) for line in f]

        if make_example in (Example.fromdict, Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)
