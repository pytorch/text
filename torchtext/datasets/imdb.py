import os
import tarfile
from six.moves import urllib
import glob

from .. import data


class IMDB(data.Dataset):

    url = 'http://ai.stanford.edu/~amaas/data/sentiment/'
    filename = 'aclImdb_v1.tar.gz'
    dirname = 'aclImdb'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the datasets highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with open(fname, 'r') as f:
                    try:
                        text = f.readline()
                    except:
                        import pdb; pdb.set_trace()
                examples.append(data.Example.fromlist([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)

    @classmethod
    def download(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            fpath = os.path.join(path, cls.filename)
            if not os.path.isfile(fpath):
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                print('downloading {}'.format(cls.filename))
                urllib.request.urlretrieve(os.path.join(cls.url, cls.filename), fpath)
            with tarfile.open(fpath, 'r:gz') as tar:
                dirs = [member for member in tar.getmembers()]
                tar.extractall(path=root, members=dirs)
        return os.path.join(path, '')


    @classmethod
    def splits(cls, text_field, label_field, root='.',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that contains the IMDB dataset subdirectory
            train: The directory that contains the training examples 
            test: The directory that contains the test examples
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
        """Creater iterator objects for splits of the IMDB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
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
