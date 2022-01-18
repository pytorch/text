from ..common.torchtext_test_case import TorchtextTestCase

from torchtext.data.datasets_utils import _ParseIOBData
from torch.utils.data.datapipes.iter import IterableWrapper


class TestDatasetUtils(TorchtextTestCase):
    def test_iob_datapipe_basic(self):
        iob = [
            "Alex I-PER",
            "is O",
            "going O",
            "to O",
            "Los I-LOC",
            "Angeles I-LOC",
            "in O",
            "California I-LOC"
        ]
        iterable = [("ignored.txt", e) for e in iob]
        iterable = IterableWrapper(iterable)
        iob_dp = list(_ParseIOBData(iterable, sep=" "))
        # There's only one example in this dataset
        self.assertEqual(len(iob_dp), 1)
        # The length of the list of surface forms is the number of lines in the example
        self.assertEqual(len(iob_dp[0][0]), len(iob))
        # The length of the list labels is the number of lines in the example
        self.assertEqual(len(iob_dp[0][1]), len(iob))
        iob = [
            "Alex I-PER",
            "is O",
            "going O",
            "to O",
            "Los I-LOC",
            "Angeles I-LOC",
            "in O",
            "California I-LOC",
            "",
            "Alex I-PER",
            "is O",
            "going O",
            "to O",
            "Los I-LOC",
            "Angeles I-LOC",
            "in O",
            "California I-LOC",
        ]
        iterable = [("ignored.txt", e) for e in iob]
        iterable = IterableWrapper(iterable)
        iob_dp = list(_ParseIOBData(iterable, sep=" "))
        # There are two examples in this dataset
        self.assertEqual(len(iob_dp), 2)
        # The length of the first list of surface forms is the length of everything before the empty line.
        # The length of the first labels is the length of everything before the empty line.
        self.assertEqual(len(iob_dp[0][0]), iob.index(""))
        self.assertEqual(len(iob_dp[0][1]), iob.index(""))
        # The length of the second list of surface forms is the length of everything after the empty line.
        # The length of the second labels is the length of everything after the empty line.
        self.assertEqual(len(iob_dp[1][0]), len(iob) - iob.index("") - 1)
        self.assertEqual(len(iob_dp[1][1]), len(iob) - iob.index("") - 1)

    def test_iob_datapipe_functional(self):
        iob = [
            "Alex I-PER",
            "is O",
            "going O",
            "to O",
            "Los I-LOC",
            "Angeles I-LOC",
            "in O",
            "California I-LOC"
        ]
        iterable = [("ignored.txt", e) for e in iob]
        iob_dp = list(IterableWrapper(iterable).read_iob(sep=" "))
        # There's only one example in this dataset
        self.assertEqual(len(iob_dp), 1)
        # The length of the list of surface forms is the number of lines in the example
        self.assertEqual(len(iob_dp[0][0]), len(iob))
        # The length of the list labels is the number of lines in the example
        self.assertEqual(len(iob_dp[0][1]), len(iob))
        iob = [
            "Alex I-PER",
            "is O",
            "going O",
            "to O",
            "Los I-LOC",
            "Angeles I-LOC",
            "in O",
            "California I-LOC",
            "",
            "Alex I-PER",
            "is O",
            "going O",
            "to O",
            "Los I-LOC",
            "Angeles I-LOC",
            "in O",
            "California I-LOC",
        ]
        iterable = [("ignored.txt", e) for e in iob]
        iob_dp = list(IterableWrapper(iterable).read_iob(sep=" "))
        # There's only one example in this dataset
        self.assertEqual(len(iob_dp), 2)
        # The length of the first list of surface forms is the length of everything before the empty line.
        # The length of the first labels is the length of everything before the empty line.
        self.assertEqual(len(iob_dp[0][0]), iob.index(""))
        self.assertEqual(len(iob_dp[0][1]), iob.index(""))
        # The length of the second list of surface forms is the length of everything after the empty line.
        # The length of the second labels is the length of everything after the empty line.
        self.assertEqual(len(iob_dp[1][0]), len(iob) - iob.index("") - 1)
        self.assertEqual(len(iob_dp[1][1]), len(iob) - iob.index("") - 1)
