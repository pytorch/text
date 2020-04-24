import torch
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

    def test_batch_iter(self):
        self.write_test_numerical_features_dataset()
        FLOAT = data.Field(use_vocab=False, sequential=False,
                           dtype=torch.float)
        INT = data.Field(use_vocab=False, sequential=False, is_target=True)
        TEXT = data.Field(sequential=False)

        dst = data.TabularDataset(path=self.test_numerical_features_dataset_path,
                                  format="tsv", skip_header=False,
                                  fields=[("float", FLOAT),
                                          ("int", INT),
                                          ("text", TEXT)])
        TEXT.build_vocab(dst)
        itr = data.Iterator(dst, batch_size=2, device=-1, shuffle=False)
        fld_order = [k for k, v in dst.fields.items() if
                     v is not None and not v.is_target]
        batch = next(iter(itr))
        (x1, x2), y = batch
        x = (x1, x2)[fld_order.index("float")]
        self.assertEquals(y.data[0], 1)
        self.assertEquals(y.data[1], 12)
        self.assertAlmostEqual(x.data[0], 0.1, places=4)
        self.assertAlmostEqual(x.data[1], 0.5, places=4)
