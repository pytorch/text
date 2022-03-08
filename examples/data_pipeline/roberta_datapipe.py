from argparse import ArgumentParser
from functools import partial
from typing import Dict, Any

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

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        tokens = self.tokenizer(input["text"])
        tokens = F.truncate(tokens, max_seq_len=254)
        tokens = self.vocab(tokens)
        tokens = self.add_bos(tokens)
        tokens = self.add_eos(tokens)
        input["tokens"] = tokens
        return input


def main(args):
    # Instantiate transform
    transform = RobertaTransform()

    # Create SST2 datapipe and apply pre-processing
    batch_size = args.batch_size
    train_dp = SST2(split="train")
    train_dp = train_dp.batch(batch_size).rows2columnar(["text", "label"])

    # Apply text pre-processing
    train_dp = train_dp.map(transform)

    # convert to Tensor
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
    parser.add_argument("--train-steps", default=-1, type=int)
    main(parser.parse_args())
