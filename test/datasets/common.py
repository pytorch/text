import pickle

from parameterized import parameterized
from torch.utils.data.graph import traverse
from torch.utils.data.graph_settings import get_all_graph_pipes
from torchdata.datapipes.iter import Shuffler, ShardingFilter
from torchtext.datasets import DATASETS

from ..common.torchtext_test_case import TorchtextTestCase


class TestDatasetPickling(TorchtextTestCase):
    @parameterized.expand(list(DATASETS.items()))
    def test_pickling(self, dataset_name, dataset_fn):
        dp = dataset_fn()
        if type(dp) == tuple:
            dp = list(dp)
        else:
            dp = [dp]

        for dp_split in dp:
            pickle.loads(pickle.dumps(dp_split))


class TestShuffleShardDatasetWrapper(TorchtextTestCase):
    # Note that for order i.e shuffle before sharding, TorchData will provide linter warning
    # Modify this test when linter warning is available
    @parameterized.expand(list(DATASETS.items()))
    def test_shuffle_shard_wrapper(self, dataset_name, dataset_fn):
        dp = dataset_fn()
        if type(dp) == tuple:
            dp = list(dp)
        else:
            dp = [dp]

        for dp_split in dp:
            dp_graph = get_all_graph_pipes(traverse(dp_split))
            for annotation_dp_type in [Shuffler, ShardingFilter]:
                if not any(isinstance(dp, annotation_dp_type) for dp in dp_graph):
                    raise AssertionError(f"The dataset doesn't contain a {annotation_dp_type.__name__}() datapipe.")
