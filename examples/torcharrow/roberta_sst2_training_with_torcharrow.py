import functools
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torcharrow._torcharrow as _ta
import torcharrow.pytorch as tap
import torchtext.functional as F
import torchtext.transforms as T
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torcharrow import functional as ta_F
from torchtext.datasets import SST2
from torchtext.models import RobertaClassificationHead, ROBERTA_BASE_ENCODER
from torchtext.utils import get_asset_local_path

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


def prepoc(df, tokenizer, vocab):
    df["tokens"] = ta_F.bpe_tokenize(tokenizer, df["text"])
    df["tokens"] = df["tokens"].list.slice(stop=254)
    df["tokens"] = ta_F.lookup_indices(vocab, df["tokens"])
    df["tokens"] = ta_F.add_tokens(df["tokens"], [0], begin=True)
    df["tokens"] = ta_F.add_tokens(df["tokens"], [2], begin=False)
    return df


def get_dataloader(split, args):
    # Instantiate TA tokenizer opaque object
    tokenizer = init_ta_gpt2bpe_encoder()

    # Instantiate TA vocab opaque object
    vocab = init_ta_gpt2bpe_vocab()

    # Create SST2 datapipe and apply pre-processing
    train_dp = SST2(split=split)

    # convert to DataFrame of size batches
    train_dp = train_dp.dataframe(columns=["text", "labels"], dataframe_size=args.batch_size)

    # Apply preproc on DataFrame
    train_dp = train_dp.map(functools.partial(prepoc, tokenizer=tokenizer, vocab=vocab))

    # (optional) Remove un-required columns
    train_dp = train_dp.map(lambda x: x.drop(["text"]))

    # convert DataFrame to tensor (This will yeild named tuple)
    train_dp = train_dp.map(lambda x: x.to_tensor({"tokens": tap.PadSequence(padding_value=1)}))

    # create DataLoader
    dl = DataLoader(train_dp, batch_size=None)

    return dl


classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
model = ROBERTA_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)


def train_step(input, target, optim, criteria):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target, criteria):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate(dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=1).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def main(args):
    print(args)
    train_dl = get_dataloader(split="train", args=args)
    dev_dl = get_dataloader(split="dev", args=args)

    learning_rate = args.learning_rate
    optim = AdamW(model.parameters(), lr=learning_rate)
    criteria = nn.CrossEntropyLoss()

    for e in range(args.num_epochs):
        for batch in train_dl:
            input = batch.tokens.to(DEVICE)
            target = batch.labels.to(DEVICE)
            train_step(input, target, optim, criteria)

    loss, accuracy = evaluate(dev_dl, criteria)
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=16, type=int, help="Input batch size used during training")
    parser.add_argument("--num-epochs", default=1, type=int, help="Number of epochs to run training")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate used for training")
    main(parser.parse_args())
