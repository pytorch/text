from __future__ import unicode_literals
import torchtext.data as data

from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    def test_batch_with_missing_field(self):
        # smoke test to see if batches with missing attributes are shown properly
        with open(self.test_missing_field_dataset_path, "wt") as f:
            f.write("text,label\n1,0")

        dst = data.TabularDataset(path=self.test_missing_field_dataset_path,
                                  format="csv", skip_header=True,
                                  fields=[("text", data.Field(use_vocab=False,
                                                              sequential=False)),
                                          ("label", None)])
        itr = data.Iterator(dst, batch_size=64)
        str(next(itr.__iter__()))
