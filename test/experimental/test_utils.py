from torchtext.experimental.functional import ngrams_func
from ..common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):
    def test_ngrams_func(self):
        func = ngrams_func(1)
        assert func(['A', 'string', 'particularly', 'one', 'with', 'slightly']) == \
            ['A', 'string', 'particularly', 'one', 'with', 'slightly']
        func = ngrams_func(2)
        assert func(['A', 'string', 'particularly', 'one', 'with', 'slightly']) == \
            ['A', 'string', 'particularly', 'one', 'with', 'slightly', 'A string', 'string particularly',
             'particularly one', 'one with', 'with slightly']
        func = ngrams_func(3)
        assert func(['A', 'string', 'particularly', 'one', 'with', 'slightly']) == \
            ['A', 'string', 'particularly', 'one', 'with', 'slightly', 'A string', 'string particularly',
             'particularly one', 'one with', 'with slightly', 'A string particularly',
             'string particularly one', 'particularly one with', 'one with slightly']
