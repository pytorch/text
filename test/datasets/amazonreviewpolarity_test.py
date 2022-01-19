#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import torchtext
import json
from parameterized import parameterized
from ..common.torchtext_test_case import TorchtextTestCase
from ..common.parameterized_utils import load_params
from ..common.case_utils import TempDirMixin
import os.path

from torchtext.datasets.amazonreviewpolarity import AmazonReviewPolarity



def get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    mocked_data = []
    sample_rate = 16000
    transcript = "This is a test transcript."

    base_dir = os.path.join(root_dir, "ARCTIC", "cmu_us_aew_arctic")
    txt_dir = os.path.join(base_dir, "etc")
    os.makedirs(txt_dir, exist_ok=True)
    txt_file = os.path.join(txt_dir, "txt.done.data")
    audio_dir = os.path.join(base_dir, "wav")
    os.makedirs(audio_dir, exist_ok=True)

    seed = 42
    with open(txt_file, "w") as txt:
        for c in ["a", "b"]:
            for i in range(5):
                utterance_id = f"arctic_{c}{i:04d}"
                path = os.path.join(audio_dir, f"{utterance_id}.wav")
                data = get_whitenoise(
                    sample_rate=sample_rate,
                    duration=3,
                    n_channels=1,
                    dtype="int16",
                    seed=seed,
                )
                save_wav(path, data, sample_rate)
                sample = (
                    normalize_wav(data),
                    sample_rate,
                    transcript,
                    utterance_id.split("_")[1],
                )
                mocked_data.append(sample)
                txt.write(f'( {utterance_id} "{transcript}" )\n')
                seed += 1
    return mocked_data


class TestAmazonReviewPolarity(TempDirMixin, TorchtextTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = get_mock_dataset(cls.root_dir)

    def _test_amazon_review_polarity(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, transcript, utterance_id) in enumerate(dataset):
            expected_sample = self.samples[i]
            assert sample_rate == expected_sample[1]
            assert transcript == expected_sample[2]
            assert utterance_id == expected_sample[3]
            self.assertEqual(expected_sample[0], waveform, atol=5e-5, rtol=1e-8)
            n_ite += 1
        assert n_ite == len(self.samples)

    def test_amazon_review_polarity_splits(self, splits):
        dataset = AmazonReviewPolarity(root=self.root_dir, split=splits)
        self._test_amazon_review_polarity(dataset)


# class TestDataset(TorchtextTestCase):
#     @classmethod
#     def setUpClass(cls):
#         check_cache_status()

#     @parameterized.expand(
#         load_params('raw_datasets.jsonl'),
#         name_func=_raw_text_custom_name_func)
#     def test_raw_text_classification(self, info):
#         dataset_name = info['dataset_name']
#         split = info['split']

#         if dataset_name == 'WMT14':
#             return
#         else:
#             data_iter = torchtext.datasets.DATASETS[dataset_name](split=split)
#         self.assertEqual(hashlib.md5(json.dumps(next(iter(data_iter)), sort_keys=True).encode('utf-8')).hexdigest(), info['first_line'])
#         if dataset_name == "AG_NEWS":
#             self.assertEqual(torchtext.datasets.URLS[dataset_name][split], info['URL'])
#             self.assertEqual(torchtext.datasets.MD5[dataset_name][split], info['MD5'])
#         elif dataset_name == "WMT14":
#             return
#         else:
#             self.assertEqual(torchtext.datasets.URLS[dataset_name], info['URL'])
#             self.assertEqual(torchtext.datasets.MD5[dataset_name], info['MD5'])
#         del data_iter

#     @parameterized.expand(list(sorted(torchtext.datasets.DATASETS.keys())))
#     def test_raw_datasets_split_argument(self, dataset_name):
#         if 'statmt' in torchtext.datasets.URLS[dataset_name]:
#             return
#         dataset = torchtext.datasets.DATASETS[dataset_name]
#         train1 = dataset(split='train')
#         train2, = dataset(split=('train',))
#         for d1, d2 in zip(train1, train2):
#             self.assertEqual(d1, d2)
#             # This test only aims to exercise the argument parsing and uses
#             # the first line as a litmus test for correctness.
#             break
#         # Exercise default constructor
#         _ = dataset()
