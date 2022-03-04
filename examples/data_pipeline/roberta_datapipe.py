from argparse import ArgumentParser
from functools import partial

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
    batch_size = args.batch_size
    train_dp = SST2(split="train")
    train_dp = train_dp.batch(batch_size).rows2columnar(["text", "label"])
    train_dp = train_dp.map(tokenizer, input_col="text", output_col="tokens")
    train_dp = train_dp.map(partial(F.truncate, max_seq_len=254), input_col="tokens")
    train_dp = train_dp.map(vocab, input_col="tokens")
    train_dp = train_dp.map(add_bos, input_col="tokens")
    train_dp = train_dp.map(add_eos, input_col="tokens")
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="tokens")
    train_dp = train_dp.map(F.to_tensor, input_col="label")

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
