from argparse import ArgumentParser
from functools import partial

import torcharrow as ta
import torcharrow.dtypes as dt
import torchtext.functional as F
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.nn import Module
from torch.utils.data import DataLoader
from torchtext.datasets import SST2


class RobertaTransform(Module):
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

    def forward(self, input: ta.IDataFrame) -> ta.IDataFrame:
        input["tokens"] = input["text"].map(self.tokenizer, dtype=dt.List(dt.string))
        input["tokens"] = input["tokens"].map(partial(F.truncate, max_seq_len=254))
        input["tokens"] = input["tokens"].map(self.vocab, dtype=dt.List(dt.int32))
        input["tokens"] = input["tokens"].map(self.add_bos)
        input["tokens"] = input["tokens"].map(self.add_eos)
        return input


def main(args):

    # transformation
    transform = RobertaTransform()

    # we need to find alternative for this for serving stack :)
    # transform_jit = torch.jit.script(transform)

    # Create SST2 datapipe and apply pre-processing
    train_dp = SST2(split="train")

    # convert to DataFrame
    train_dp = train_dp.dataframe(columns=["text", "label"], dataframe_size=args.dataframe_size)

    # Apply transformation on DataFrame
    train_dp = train_dp.map(transform)

    # keep necessary columns
    train_dp = train_dp.map(lambda x: x["tokens", "label"])

    train_dp = train_dp.map(lambda x: x.batch(args.batch_size))

    # create DataLoader
    dl = DataLoader(train_dp, batch_size=None)

    train_steps = args.train_steps
    for i, batch in enumerate(dl):
        if i == train_steps:
            break

        # model_input = batch["tokens"]
        # target = batch["label"]
        ...


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--train-steps", default=-1, type=int)
    parser.add_argument("--dataframe-size", default=100, type=int)
    main(parser.parse_args())
