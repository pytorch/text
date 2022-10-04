import pickle

from parameterized import parameterized
from torch.utils.data.graph import traverse_dps
from torch.utils.data.graph_settings import get_all_graph_pipes
from torchdata.dataloader2.linter import _check_shuffle_before_sharding
from torchdata.datapipes.iter import Shuffler, ShardingFilter
from torchtext.datasets import DATASETS

from ..common.torchtext_test_case import TorchtextTestCase


class TestDatasetPickling(TorchtextTestCase):
    @parameterized.expand([(f,) for f in DATASETS.values()])
    def test_pickling(self, dataset_fn):
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
    @parameterized.expand([(f,) for f in DATASETS.values()])
    def test_shuffle_shard_wrapper(self, dataset_fn):
        dp = dataset_fn()
        if type(dp) == tuple:
            dp = list(dp)
        else:
            dp = [dp]

        for dp_split in dp:
            _check_shuffle_before_sharding(dp_split)

            dp_graph = get_all_graph_pipes(traverse_dps(dp_split))
            for annotation_dp_type in [Shuffler, ShardingFilter]:
                if not any(isinstance(dp, annotation_dp_type) for dp in dp_graph):
                    raise AssertionError(f"The dataset doesn't contain a {annotation_dp_type.__name__}() datapipe.")
