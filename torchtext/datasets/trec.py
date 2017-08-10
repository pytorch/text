import os

from .. import data
from six.moves import urllib
from bs4 import UnicodeDammit


class TREC(data.ZipDataset):

    url_base = 'http://cogcomp.org/Data/QA/QC/'
    train_filename = 'train_5500.label'
    test_filename = 'TREC_10.label'
    dirname = 'trec'

    def __init__(self, path, text_field, label_field,
                 fine_grained=False, **kwargs)
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            fine_grained: Whether to use the fine-grained (50-class) version of TREC
                or the coarse grained (6-class) version.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        def get_label_str(label):
                return label.split(':')[0] if not fine_grained else label
        label_field.preprocessing = data.Pipeline(get_label_str)

        for line in open(os.path.expanduser(path), 'rb'):
            line = UnicodeDammit(line).unicode_markup.split()
            label, text = line[0], ' '.join(line[1:])
            examples.append(data.Example.fromlist([text, label], fields))

        super(TREC, self).__init__(examples, fields, **kwargs)

    @classmethod
    def download(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            for fn in [cls.train_filename, cls.test_filename]:
                fpath = os.path.join(path, fn)
                if not os.path.isfile(fpath):
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    print('downloading {}'.format(fn))
                    urllib.request.urlretrieve(os.path.join(cls.url_base, fn), fpath)
        return os.path.join(path, '')

    @classmethod
    def splits(cls, text_field, label_field, root='.',
               train=train_filename, test=test_filename, **kwargs):
        """Create dataset objects for splits of the TREC dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train_5500.label'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'TREC_10.label'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download(root)

        train_data = None if train is None else cls(
            path + train, text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            path + test, text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_dir='.',
              wv_type=None, wv_dim='300d', **kwargs):
        """Creater iterator objects for splits of the TREC dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
