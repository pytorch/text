import json
from argparse import ArgumentParser
from functools import partial
from typing import Dict, Any

import torch
import torcharrow as ta
import torcharrow._torcharrow as _ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
import torchtext.functional as F
import torchtext.transforms as T
from benchmark.utils import Timer
from torch.hub import load_state_dict_from_url
from torch.nn import Module
from torcharrow import functional as ta_F
from torchtext.datasets import DATASETS
from torchtext.utils import get_asset_local_path


class RobertaTransformDataPipe(Module):
    def __init__(self) -> None:
        super().__init__()
        # Instantiate various transforms

        # Tokenizer to split input text into tokens
        encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
        vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
        self.tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)

        # vocabulary converting tokens to IDs
        vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
        self.vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))

        # Add BOS token to the beginning of sentence
        self.add_bos = T.AddToken(token=0, begin=True)

        # Add EOS token to the end of sentence
        self.add_eos = T.AddToken(token=2, begin=False)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        tokens = self.tokenizer(input["text"])
        tokens = F.truncate(tokens, max_seq_len=254)
        tokens = self.vocab(tokens)
        tokens = self.add_bos(tokens)
        tokens = self.add_eos(tokens)
        input["tokens"] = tokens
        return input


def init_ta_gpt2bpe_encoder():
    encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"

    encoder_json_path = get_asset_local_path(encoder_json_path)
    vocab_bpe_path = get_asset_local_path(vocab_bpe_path)
    _seperator = "\u0001"

    # load bpe encoder and bpe decoder
    with open(encoder_json_path, "r", encoding="utf-8") as f:
        bpe_encoder = json.load(f)
    # load bpe vocab
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_vocab = f.read()
    bpe_merge_ranks = {
        _seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
    }
    # Caching is enabled in Eager mode
    bpe = _ta.GPT2BPEEncoder(bpe_encoder, bpe_merge_ranks, _seperator, T.bytes_to_unicode(), True)
    return bpe


def init_ta_gpt2bpe_vocab():
    vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
    vocab_path = get_asset_local_path(vocab_path)
    vocab = torch.load(vocab_path)
    ta_vocab = _ta.Vocab(vocab.get_itos(), vocab.get_default_index())
    return ta_vocab


class RobertaTransformDataFrameNativeOps(Module):
    def __init__(self) -> None:
        super().__init__()
        # Tokenizer to split input text into tokens
        self.tokenizer = init_ta_gpt2bpe_encoder()

        # vocabulary converting tokens to IDs
        self.vocab = init_ta_gpt2bpe_vocab()

        # Add BOS token to the beginning of sentence
        self.add_bos = T.AddToken(token=0, begin=True)

        # Add EOS token to the end of sentence
        self.add_eos = T.AddToken(token=2, begin=False)

    def forward(self, input: ta.DataFrame) -> ta.DataFrame:
        input["tokens"] = ta_F.bpe_tokenize(self.tokenizer, input["text"])
        input["tokens"] = input["tokens"].list.slice(stop=254)
        input["tokens"] = ta_F.lookup_indices(self.vocab, input["tokens"])
        input["tokens"] = input["tokens"].transform(self.add_bos, format="python")
        input["tokens"] = input["tokens"].transform(self.add_eos, format="python")
        return input


class RobertaTransformDataFrameUDF(Module):
    def __init__(self) -> None:
        super().__init__()
        # Instantiate various transforms

        # Tokenizer to split input text into tokens
        encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
        vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
        self.tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)

        # vocabulary converting tokens to IDs
        vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
        self.vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))

        # Add BOS token to the beginning of sentence
        self.add_bos = T.AddToken(token=0, begin=True)

        # Add EOS token to the end of sentence
        self.add_eos = T.AddToken(token=2, begin=False)

    def forward(self, input: ta.DataFrame) -> ta.DataFrame:
        input["tokens"] = input["text"].transform(self.tokenizer, dtype=dt.List(dt.string), format="python")
        input["tokens"] = input["tokens"].list.slice(stop=254)
        input["tokens"] = input["tokens"].transform(self.vocab, dtype=dt.List(dt.int32), format="python")
        input["tokens"] = input["tokens"].transform(self.add_bos, format="python")
        input["tokens"] = input["tokens"].transform(self.add_eos, format="python")
        return input


def benchmark_roberta_datapipe(args):
    print("********Running Benchmark using DataPipes**************\n")
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    columns = args.columns

    # Instantiate transform
    with Timer("Initialize Roberta Transform (for datapipe)"):
        transform = RobertaTransformDataPipe()

    with Timer("Initialize Pipeline"):
        # Create SST2 datapipe and apply pre-processing
        train_dp = DATASETS[dataset_name](split="train")
        train_dp = train_dp.batch(batch_size).rows2columnar(columns)

        # Apply text pre-processing
        train_dp = train_dp.map(transform)

        # convert to Tensor
        train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="tokens")
        train_dp = train_dp.map(F.to_tensor, input_col="label")

    with Timer("Execute Pipeline"):
        list(train_dp)


def benchmark_roberta_dataframe(args, native_ops):
    print("****************Running Benchmark using TorchArrow Dataframes*********************\n")
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    columns = args.columns

    if native_ops:
        append_text = "as native ops"
    else:
        append_text = "as UDF"

    # Instantiate transform
    with Timer("Initialize Roberta Transform (for DataFrame) {}".format(append_text)):
        if native_ops:
            transform = RobertaTransformDataFrameNativeOps()
        else:
            transform = RobertaTransformDataFrameUDF()

    with Timer("Initialize Pipeline"):
        # Create SST2 datapipe and apply pre-processing
        train_dp = DATASETS[dataset_name](split="train")

        # convert to DataFrame of size batches
        # TODO: Figure out how to create DataFrame of larger size and create smaller batches
        train_dp = train_dp.dataframe(columns=columns, dataframe_size=batch_size)

        # Apply transformation on DataFrame
        train_dp = train_dp.map(transform)

        # Remove not required columns
        train_dp = train_dp.map(lambda x: x.drop(["text"]))

        # convert DataFrame to tensor (This will yeild named tuple)
        train_dp = train_dp.map(lambda x: x.to_tensor({"tokens": tap.PadSequence(padding_value=1)}))

    with Timer("Execute Pipeline"):
        list(train_dp)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset-name", default="SST2", type=str)
    parser.add_argument("--columns", default=["text", "label"], nargs="+")
    benchmark_roberta_datapipe(parser.parse_args())
    benchmark_roberta_dataframe(parser.parse_args(), native_ops=False)
    benchmark_roberta_dataframe(parser.parse_args(), native_ops=True)
