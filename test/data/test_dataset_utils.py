from parameterized import parameterized
from torch.utils.data.datapipes.iter import IterableWrapper
from torchtext.data.datasets_utils import _ParseIOBData

from ..common.torchtext_test_case import TorchtextTestCase


class TestDatasetUtils(TorchtextTestCase):
    @parameterized.expand(
        [
            [lambda it: list(_ParseIOBData(IterableWrapper(it), sep=" "))],
            [lambda it: list(IterableWrapper(it).read_iob(sep=" "))],
        ]
    )
    def test_iob_datapipe(self, pipe_fn):
        iob = ["Alex I-PER", "is O", "going O", "to O", "Los I-LOC", "Angeles I-LOC", "in O", "California I-LOC"]
        iterable = [("ignored.txt", e) for e in iob]
        iob_dp = pipe_fn(iterable)
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
        iob_dp = pipe_fn(iterable)
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
