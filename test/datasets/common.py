import pickle
from functools import partial
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized

from ..common.case_utils import TempDirMixin, zip_equal
from ..common.torchtext_test_case import TorchtextTestCase
from . import (
    test_agnews,
    test_amazonreviews,
    test_cc100,
    test_conll2000chunking,
    test_dbpedia,
    test_enwik9,
    test_imdb,
    test_iwslt2016,
    test_iwslt2017,
    test_multi30k,
    test_penntreebank,
    test_sogounews,
    test_squads,
    test_sst2,
    test_udpos,
    test_wikitexts,
    test_yahooanswers,
    test_yelpreviews,
)

SPLIT = "train"
TEST_PARAMETERIZED_ARGS = [
    (test_agnews.AG_NEWS, test_agnews._get_mock_dataset),
    (
        test_amazonreviews.AmazonReviewFull,
        partial(test_amazonreviews._get_mock_dataset, base_dir_name=test_amazonreviews.AmazonReviewFull.__name__),
    ),
    (
        test_amazonreviews.AmazonReviewPolarity,
        partial(test_amazonreviews._get_mock_dataset, base_dir_name=test_amazonreviews.AmazonReviewPolarity.__name__),
    ),
    (test_cc100.CC100, test_cc100._get_mock_dataset),
    (test_conll2000chunking.CoNLL2000Chunking, test_conll2000chunking._get_mock_dataset),
    (test_dbpedia.DBpedia, test_dbpedia._get_mock_dataset),
    (test_enwik9.EnWik9, test_enwik9._get_mock_dataset),
    (test_imdb.IMDB, test_imdb._get_mock_dataset),
    (test_iwslt2016.IWSLT2016, test_iwslt2016._get_mock_dataset),
    (test_iwslt2017.IWSLT2017, test_iwslt2017._get_mock_dataset),
    (test_multi30k.Multi30k, test_multi30k._get_mock_dataset),
    (test_penntreebank.PennTreebank, test_penntreebank._get_mock_dataset),
    (test_sogounews.SogouNews, test_sogounews._get_mock_dataset),
    (test_squads.SQuAD1, partial(test_squads._get_mock_dataset, base_dir_name=test_squads.SQuAD1.__name__)),
    (test_squads.SQuAD2, partial(test_squads._get_mock_dataset, base_dir_name=test_squads.SQuAD2.__name__)),
    (test_sst2.SST2, test_sst2._get_mock_dataset),
    (test_udpos.UDPOS, test_udpos._get_mock_dataset),
    (
        test_wikitexts.WikiText2,
        partial(test_wikitexts._get_mock_dataset, base_dir_name=test_wikitexts.WikiText2.__name__),
    ),
    (
        test_wikitexts.WikiText103,
        partial(test_wikitexts._get_mock_dataset, base_dir_name=test_wikitexts.WikiText103.__name__),
    ),
    (test_yahooanswers.YahooAnswers, test_yahooanswers._get_mock_dataset),
    (
        test_yelpreviews.YelpReviewFull,
        partial(test_yelpreviews._get_mock_dataset, base_dir_name=test_yelpreviews.YelpReviewFull.__name__),
    ),
    (
        test_yelpreviews.YelpReviewPolarity,
        partial(test_yelpreviews._get_mock_dataset, base_dir_name=test_yelpreviews.YelpReviewPolarity.__name__),
    ),
]


def _generate_mock_dataset(dataset_fn, get_mock_dataset_fn, root_dir):
    if dataset_fn == test_cc100.CC100:
        language_code = "en"

        expected_samples = list(get_mock_dataset_fn(root_dir)[language_code])
        dp = dataset_fn(root=root_dir, language_code=language_code)
    elif dataset_fn == test_enwik9.EnWik9:
        expected_samples = list(get_mock_dataset_fn(root_dir))
        dp = dataset_fn(root=root_dir)
    elif dataset_fn == test_iwslt2016.IWSLT2016:
        language_pair = ("de", "en")
        valid_set = "tst2013"
        test_set = "tst2014"

        expected_samples = get_mock_dataset_fn(root_dir, SPLIT, language_pair[0], language_pair[1], valid_set, test_set)
        dp = dataset_fn(
            root=root_dir,
            split=SPLIT,
            language_pair=language_pair,
            valid_set=valid_set,
            test_set=test_set,
        )
    elif dataset_fn == test_iwslt2017.IWSLT2017:
        language_pair = ("de", "en")
        valid_set = "dev2010"
        test_set = "tst2010"

        expected_samples = get_mock_dataset_fn(root_dir, SPLIT, language_pair[0], language_pair[1], valid_set, test_set)
        dp = dataset_fn(root=root_dir, split=SPLIT, language_pair=language_pair)
    elif dataset_fn == test_multi30k.Multi30k:
        language_pair = ("en", "de")
        expected_samples = get_mock_dataset_fn(root_dir)
        expected_samples = [
            (d1, d2)
            for d1, d2 in zip(
                expected_samples[f"{SPLIT}.{language_pair[0]}"],
                expected_samples[f"{SPLIT}.{language_pair[1]}"],
            )
        ]
        dp = dataset_fn(root=root_dir, split=SPLIT, language_pair=language_pair)
    else:
        expected_samples = list(get_mock_dataset_fn(root_dir)[SPLIT])
        dp = dataset_fn(root=root_dir, split=SPLIT)

    return expected_samples, dp


class TestDatasetPickling(TempDirMixin, TorchtextTestCase):
    root_dir = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.root_dir = cls.get_base_temp_dir()
        cls.patcher = patch("torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        super().tearDownClass()

    @parameterized.expand(TEST_PARAMETERIZED_ARGS)
    def test_pickling(self, dataset_fn, get_mock_dataset_fn):
        expected_samples, dp1 = _generate_mock_dataset(dataset_fn, get_mock_dataset_fn, self.root_dir)
        pickle_file = (Path(self.root_dir) / (dataset_fn.__name__ + ".pkl")).resolve()
        pickle.dump(dp1, open(pickle_file, "wb"))
        dp2 = pickle.load(open(pickle_file, "rb"))

        samples = list(dp2)
        for sample, expected_sample in zip_equal(samples, expected_samples):
            self.assertEqual(sample, expected_sample)
