import torchtext.data as data

from ..common.torchtext_test_case import TorchtextTestCase


class TestIterator(TorchtextTestCase):

    def test_batch(self):
        elements = [2, 5, 6, 4, 8, 1, 2]
        generator = data.batch(elements, 3)
        assert next(generator) == [2, 5, 6]
        assert next(generator) == [4, 8, 1]
        assert next(generator) == [2]

    def test_custom_batch_fn(self):
        def _accu(element, count, size_so_far):
            return size_so_far + element

        elements = [2, 5, 6, 4, 8, 1, 2]
        generator = data.batch(elements, 10, batch_size_fn=_accu)
        assert next(generator) == [2, 5]
        assert next(generator) == [6, 4]
        assert next(generator) == [8, 1]
        assert next(generator) == [2]

    def test_custom_batch_fn_multiple(self):
        def _accu(element, count, size_so_far):
            return size_so_far + element

        elements = [2, 5, 7, 3, 1, 6, 4, 2, 8, 1, 2]
        generator = data.batch(
            elements, 20, batch_size_fn=_accu, batch_size_multiple=3)
        assert next(generator) == [2, 5, 7]
        assert next(generator) == [3, 1, 6]
        assert next(generator) == [4, 2, 8, 1, 2]
