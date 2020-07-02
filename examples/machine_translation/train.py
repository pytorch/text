import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.data import DataLoader

from dataset import get_dataset
from embedding import WordCharCNNEmbedding
from model import Attention, Decoder, Encoder, Seq2Seq
from utils import collate_char_fn, count_parameters, epoch_time


def train(model: nn.Module, iterator: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, clip: float):
    model.train()

    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]
        optimizer.zero_grad()

        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]

            output = model(src, trg, 0)  # turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    train_dataset, val_dataset, test_dataset = get_dataset()
    train_iterator = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_char_fn)
    valid_iterator = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_char_fn)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_char_fn)
    input_dim = len(train_dataset.vocab)
    output_dim = len(train_dataset.vocab)

    # enc_emb_dim = 256
    # dec_emb_dim = 256
    # enc_hid_dim = 512
    # dec_hid_dim = 512
    # attn_dim = 64
    # enc_dropout = 0.5
    # dec_dropout = 0.5

    enc_emb_dim = 32
    dec_emb_dim = 32
    enc_hid_dim = 64
    dec_hid_dim = 64
    attn_dim = 8
    enc_dropout = 0.5
    dec_dropout = 0.5

    emb = WordCharCNNEmbedding(input_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout, emb)
    attn = Attention(enc_hid_dim, dec_hid_dim, attn_dim)
    dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn, emb)
    model = Seq2Seq(enc, dec, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters())
    print(f"The model has {count_parameters(model):,} trainable parameters")
    n_epochs = 10
    clip = 1

    best_valid_loss = float("inf")

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")

    test_loss = evaluate(model, test_iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
