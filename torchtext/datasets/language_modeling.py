import torch

import itertools
import math
import os

from .. import data


class LanguageModelingDataset(data.Dataset):

    def __init__(self, path, fields, newline_eos=True, **kwargs):

        if not isinstance(fields, (tuple, list)):
            fields = [('text', fields)]

        field = fields[0][1]

        path = os.path.expanduser(path)

        text = []
        with open(path) as f:
            for line in f:
                text += field.preprocess(line)
                if newline_eos:
                    text.append('<eos>')

        examples = [data.Example.fromlist([text], fields)]

        # chunks = itertools.zip_longest(*[iter(text)] * field.fix_length,
        #                                fillvalue='<pad>')
        # target_chunks = itertools.zip_longest(
        #     *[iter(text[1:] + ['<pad>'])] * field.fix_length, fillvalue='<pad>')

        # examples = [data.Example.fromlist([chunk, target_chunk], fields)
        #             for chunk, target_chunk in zip(chunks, target_chunks)]

        super().__init__(examples, fields, **kwargs)


class WikiText2(LanguageModelingDataset, data.ZipDataset):

    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    filename = 'wikitext-2-v1.zip'
    dirname = 'wikitext-2'

    @classmethod
    def splits(cls, field, root='.', train='train.tokens',
               validation='valid.tokens', test='test.tokens'):
        path = cls.download_or_unzip(root)
        return super().splits(os.path.join(path, 'wiki.'), train, validation,
                              test, fields=field)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.',
              wv_path=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, wv_path=wv_path)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)
