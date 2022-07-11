import json
from argparse import ArgumentParser

import torch
import torcharrow as ta
import torcharrow._torcharrow as _ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.nn import Module
from torch.utils.data import DataLoader
from torcharrow import functional as ta_F
from torchtext.datasets import SST2
from torchtext.utils import get_asset_local_path


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
        input["tokens"] = ta_F.add_tokens(input["tokens"], [0], begin=True)
        input["tokens"] = ta_F.add_tokens(input["tokens"], [2], begin=False)
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


def main(args):

    # Instantiate transform
    if args.ops_type == "udf":
        transform = RobertaTransformDataFrameUDF()
    elif args.ops_type == "native":
        transform = RobertaTransformDataFrameNativeOps()
    else:
        raise Exception("Wrong ops type provided. Available options are `udf` and `native`")

    # Create SST2 datapipe and apply pre-processing
    train_dp = SST2(split="train")

    # convert to DataFrame of size batches
    # TODO: Figure out how to create DataFrame of larger size and create batches consequently
    train_dp = train_dp.dataframe(columns=["text", "labels"], dataframe_size=args.batch_size)

    # Apply transformation on DataFrame
    train_dp = train_dp.map(transform)

    # Remove not required columns
    train_dp = train_dp.map(lambda x: x.drop(["text"]))

    # convert DataFrame to tensor (This will yeild named tuple)
    train_dp = train_dp.map(lambda x: x.to_tensor({"tokens": tap.PadSequence(padding_value=1)}))

    # create DataLoader
    dl = DataLoader(train_dp, batch_size=None)

    num_steps = args.num_steps
    for i, batch in enumerate(dl):
        if i == num_steps:
            break

        # model_input = batch.tokens
        # target = batch.labels
        ...


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-steps", default=-1, type=int)
    parser.add_argument("--ops-type", default="udf", choices=["udf", "native"], type=str)
    main(parser.parse_args())
