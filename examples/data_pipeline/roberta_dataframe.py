from argparse import ArgumentParser
from functools import partial

import torcharrow as ta
import torcharrow.dtypes as dt
import torchtext.functional as F
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchtext.datasets import SST2


def main(args):
    # Instantiate various transforms

    # Tokenizer to split input text into tokens
    encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
    tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)

    # vocabulary converting tokens to IDs
    vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
    vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))

    # Add BOS token to the beginning of sentence
    add_bos = T.AddToken(token=0, begin=True)

    # Add EOS token to the end of sentence
    add_eos = T.AddToken(token=2, begin=False)

    # Convert to torch Tensor

    # Create SST2 datapipe and apply pre-processing
    train_dp = SST2(split="train")
    data = list(train_dp)
    # Creating dataframe takes forever for ~67k items in the list
    train_df = ta.DataFrame(data, dtype=dt.Struct([dt.Field("text", dt.string), dt.Field("label", dt.int32)]))
    train_df["tokens"] = train_df["text"].map(tokenizer, dtype=dt.List(dt.string))
    train_df["tokens"] = train_df["tokens"].map(partial(F.truncate, max_seq_len=254))
    train_df["tokens"] = train_df["tokens"].map(vocab, dtype=dt.List(dt.int32))
    train_df["tokens"] = train_df["tokens"].map(add_bos)
    train_df["tokens"] = train_df["tokens"].map(add_eos)

    # TODO: Convert Dataframe back to datapipe
    # train_dp = train_df.collate(...)

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
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--train_steps", default=-1, type=int)
    main(parser.parse_args())
