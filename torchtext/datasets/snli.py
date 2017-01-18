import os

from .. import data

class SNLI(data.ZipDataset, data.TabularDataset):

    url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    filename = 'snli_1.0.zip'
    dirname = 'snli_1.0'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, root='.', trees=False,
               train='train.jsonl', validation='dev.jsonl', test='test.jsonl'):
        if trees:
            raise NotImplementedError
        path = cls.download_or_unzip(root)
        return super().splits(os.path.join(path, 'snli_1.0_'),
                              train, validation, test,
                              format='json', fields={
                                'sentence1': ('premise', text_field),
                                'sentence2': ('hypothesis', text_field),
                                'gold_label': ('label', label_field)},
                              filter_pred=lambda ex: ex.label != '-')

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_path=None, **kwargs):
        TEXT = data.Field(tokenize='spacy')
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, wv_path=wv_path)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
