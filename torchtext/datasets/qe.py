import os

from .. import data


class QualityEstimationDataset(data.Dataset):
    """Defines a dataset for quality estimation. Based on the WMT 2018."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a QualityEstimationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]),
                      ('src_tag_path', fields[2]), ('trg_tag_path', fields[3])]

        src_path, trg_path, src_tag_path, trg_tag_path = tuple(
            os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, \
                open(trg_path) as trg_file, \
                open(src_tag_path) as src_tag_file, \
                open(trg_tag_path) as trg_tag_file:
            for src_line, trg_line, src_tag_line, trg_tag_line \
                    in zip(src_file, trg_file, src_tag_file, trg_tag_file):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                src_tag_line = src_tag_line.strip()
                trg_tag_line = trg_tag_line.strip()
                if src_line != '' and trg_line != '' and \
                        src_tag_line != '' and trg_tag_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line,
                         src_tag_line, trg_tag_line], fields))

        super(QualityEstimationDataset, self).__init__(
            examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='dev', test='test', **kwargs):
        """Create dataset objects for splits of a QualityEstimationDataset.

        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
