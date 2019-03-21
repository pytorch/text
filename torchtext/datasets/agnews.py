import io
import os
import tarfile

from .. import data
from ..data.example import Example
from ..utils import unicode_csv_reader, download_file_from_google_drive

class AGNEWS(data.Dataset):

    file_id = '0Bz8a_Dbh9QhbUDNpeUdjb0wxRms'
    file_hash = '3fbb43ee5cdb9b9ac5cac71bf7f75ecc'
    name = 'agnews'
    dirname = 'ag_news_csv'

    @staticmethod
    def sort_key(ex):
        return len(ex.description)

    def __init__(self, path, title_field, description_field, label_field, **kwargs):
        
        fields = [('label', label_field), ('title', title_field), ('description', description_field)]
        
        with io.open(os.path.expanduser(path), encoding="utf8") as f:

            reader = unicode_csv_reader(f, delimiter=',')

            examples = [Example.fromCSV(line, fields) for line in reader]

        super(AGNEWS, self).__init__(examples, fields, *kwargs)

    @classmethod
    def splits(cls, title_field, description_field, label_field, root='.data', train='ag_news_csv/train.csv', test='ag_news_csv/test.csv', **kwargs):
        
        download_file_from_google_drive('0Bz8a_Dbh9QhbUDNpeUdjb0wxRms', root, 'ag_news_csv.tar.gz', '3fbb43ee5cdb9b9ac5cac71bf7f75ecc') 

        tar = tarfile.open(os.path.join(root, 'ag_news_csv.tar.gz'))
        tar.extractall(root)
        tar.close()

        train_data = None if train is None else cls(
            os.path.join(root, train), title_field, description_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(root, test), title_field, description_field, label_field, **kwargs)
        return tuple(d for d in (train_data, test_data) if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=None, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the IMDB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)

            Remaining keyword arguments: Passed to the splits method.
        """
        TITLE = data.Field()
        DESCRIPTION = data.Field()
        LABEL = data.LabelField()

        train, test = cls.splits(TITLE, DESCRIPTION, LABEL, root=root, **kwargs)

        TITLE.build_vocab(train, vectors=vectors)
        DESCRIPTION.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
        