from argparse import ArgumentParser

import torcharrow as ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
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

    def forward(self, input: ta.DataFrame) -> ta.DataFrame:
        input["tokens"] = input["text"].map(self.tokenizer.forward, dtype=dt.List(dt.string))
        input["tokens"] = input["tokens"].list.slice(stop=254)
        input["tokens"] = input["tokens"].map(self.vocab, dtype=dt.List(dt.int32))
        input["tokens"] = input["tokens"].map(self.add_bos)
        input["tokens"] = input["tokens"].map(self.add_eos)
        return input


def main(args):

    # Instantiate transform
    transform = RobertaTransform()

    # Create SST2 datapipe and apply pre-processing
    train_dp = SST2(split="train")

    # convert to DataFrame of size batches
    # TODO: Figure out how to create DataFrame of larger size and create batches consequently
    train_dp = train_dp.dataframe(columns=["text", "labels"], dataframe_size=args.batch_size)

    for batch in train_dp:
        raise Exception(batch)

    # Apply transformation on DataFrame
    train_dp = train_dp.map(transform)

    # Remove not required columns
    train_dp = train_dp.map(lambda x: x.drop(["text"]))

    # convert DataFrame to tensor (This will yeild named tuple)
    train_dp = train_dp.map(lambda x: x.to_tensor({"tokens": tap.PadSequence(padding_value=1)}))

    # create DataLoader
    dl = DataLoader(train_dp, batch_size=None)

    train_steps = args.train_steps
    for i, batch in enumerate(dl):
        if i == train_steps:
            break

        # model_input = batch.tokens
        # target = batch.labels
        ...


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--train-steps", default=-1, type=int)
    parser.add_argument("--dataframe-size", default=100, type=int)
    main(parser.parse_args())
