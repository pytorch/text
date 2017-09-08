# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import six
import torchtext.data as data

from ..common.torchtext_test_case import TorchtextTestCase


class TestPipeline(TorchtextTestCase):
    @staticmethod
    def repeat_n(x, n=3):
        """
        Given a sequence, repeat it n times.
        """
        return x * n

    def test_pipeline(self):
        pipeline = data.Pipeline(six.text_type.lower)
        assert pipeline("Test STring") == "test string"
        assert pipeline("ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T") == "ᑌᑎiᑕoᗪᕮ_tᕮ᙭t"
        assert pipeline(["1241", "Some String"]) == ["1241", "some string"]

        args_pipeline = data.Pipeline(TestPipeline.repeat_n)
        assert args_pipeline("test", 5) == "testtesttesttesttest"
        assert args_pipeline(["ele1", "ele2"], 2) == ["ele1ele1", "ele2ele2"]

    def test_composition(self):
        pipeline = data.Pipeline(TestPipeline.repeat_n)
        pipeline.add_before(six.text_type.lower)
        pipeline.add_after(six.text_type.capitalize)

        other_pipeline = data.Pipeline(six.text_type.swapcase)
        other_pipeline.add_before(pipeline)

        # Assert pipeline gives proper results after composition
        # (test that we aren't modfifying pipes member)
        assert pipeline("teST") == "Testtesttest"
        assert pipeline(["ElE1", "eLe2"]) == ["Ele1ele1ele1", "Ele2ele2ele2"]

        # Assert pipeline that we added to gives proper results
        assert other_pipeline("teST") == "tESTTESTTEST"
        assert other_pipeline(["ElE1", "eLe2"]) == ["eLE1ELE1ELE1", "eLE2ELE2ELE2"]

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            data.Pipeline("Not Callable")
