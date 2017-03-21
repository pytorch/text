import os

from .. import data


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    sort_field = 'src'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.tgt))


    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('tgt', fields[1])]

        src_path, tgt_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(tgt_path) as tgt_file:
            for src_line, tgt_line in zip(src_file, tgt_file):
                src_line, tgt_line = src_line.strip(), tgt_line.strip()
                if src_line != '' and tgt_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, tgt_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)
