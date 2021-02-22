import time
import argparse
from torchtext.experimental.transforms import load_sp_model as load_pybind_sp_model
from torchtext.data.functional import load_sp_model as load_torchbind_sp_model
from torchtext.utils import download_from_url
from torchtext.datasets import text_classification as raw


def benchmark_sentencepiece(args):
    def _run_benchmark(train, spm_processor):
        t0 = time.monotonic()
        for (_, text) in train:
            spm_processor(text)
        print("Sentencepiece processor time:", time.monotonic() - t0)

    # Download a pretrained sentencepiece model
    sp_model_path = download_from_url('https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_15000.model')

    # existing sentencepiece model with torchbind
    train, _ = raw.DATASETS[args.dataset]()
    sp_model = load_torchbind_sp_model(sp_model_path)
    print("SentencePiece EncodeAsIds - torchbind")
    _run_benchmark(train, sp_model.EncodeAsIds)

    # experimental sentencepiece model with pybind
    train, _ = raw.DATASETS[args.dataset]()
    sp_model = load_pybind_sp_model(sp_model_path)
    print("SentencePiece EncodeAsIds - pybind")
    _run_benchmark(train, sp_model.EncodeAsIds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SentencePiece benchmark')
    parser.add_argument('--dataset', type=str, default='AG_NEWS',
                        help='Dataset for performance benchmark')
    args = parser.parse_args()
    benchmark_sentencepiece(args)

# Running with AG_NEWS
# SentencePiece EncodeAsIds - torchbind
# Sentencepiece processor time: 11.536989663727582
# SentencePiece EncodeAsIds - pybind
# Sentencepiece processor time: 11.38821320142597

# Running with YelpReviewFull
# SentencePiece EncodeAsIds - torchbind
# Sentencepiece processor time: 224.23954573180526
# SentencePiece EncodeAsIds - pybind
# Sentencepiece processor time: 217.134037473239
