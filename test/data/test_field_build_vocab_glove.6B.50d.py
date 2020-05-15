import torch.nn

import torchtext.data as data
from ..common.torchtext_test_case import TorchtextTestCase


class TestNestedField(TorchtextTestCase):
    def test_build_vocab(self):
        # This test requires network access
        nesting_field = data.Field(tokenize=list, init_token="<w>", eos_token="</w>")

        field = data.NestedField(nesting_field, init_token='<s>', eos_token='</s>',
                                 include_lengths=True,
                                 pad_first=True)

        sources = [[['a'], ['s', 'e', 'n', 't', 'e', 'n', 'c', 'e'], ['o', 'f'],
                    ['d', 'a', 't', 'a'], ['.']],
                   [['y', 'e', 't'], ['a', 'n', 'o', 't', 'h', 'e', 'r']],
                   [['o', 'n', 'e'], ['l', 'a', 's', 't'], ['s', 'e', 'n', 't']]]

        field.build_vocab(sources, vectors='glove.6B.50d',
                          unk_init=torch.nn.init.normal_,
                          vectors_cache=".vector_cache")
