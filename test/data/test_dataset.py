# -*- coding: utf-8 -*-
import torchtext.data as data

from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    def test_tabular_csv(self):
        self.write_csv_dataset()
        question_field = data.Field(sequential=True)
        label_field = data.Field(sequential=False)
        dataset = data.TabularDataset(
            path=self.csv_dataset_path, format="CsV",
            fields=[("id", None), ("q1", question_field),
                    ("q2", question_field), ("label", label_field)])

        assert len(dataset) == 3

        expected_examples = [
            (["When", "do", "you", "use", "シ", "instead", "of", "し?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"], "1"),
            (["What", "is", "2+2"], ["2+2=?"], "1")]

        # Ensure examples have correct contents / test __getitem__
        for i in range(len(dataset)):
            self.assertEqual(dataset[i].q1, expected_examples[i][0])
            self.assertEqual(dataset[i].q2, expected_examples[i][1])
            self.assertEqual(dataset[i].label, expected_examples[i][2])

        # Test __getattr__
        for i, (q1, q2, label) in enumerate(zip(dataset.q1, dataset.q2,
                                                dataset.label)):
            self.assertEqual(q1, expected_examples[i][0])
            self.assertEqual(q2, expected_examples[i][1])
            self.assertEqual(label, expected_examples[i][2])

        # Test __iter__
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q2, expected_examples[i][1])
            self.assertEqual(example.label, expected_examples[i][2])
