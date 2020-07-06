import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from embedding import WordCharCNNEmbedding
from model import Attention, Decoder, Encoder, Seq2Seq
from torchtext.data.metrics import bleu_score
from torchtext.vocab import Vocab
from utils import collate_fn, count_parameters, epoch_time, seed_everything

# Ensure reproducibility
seed_everything(42)


def train(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    trg_vocab: Vocab,
    device: torch.device,
):
    model.train()

    epoch_loss = 0
    bleu = 0.0
    for _, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # Need to convert to [seq_len x batch x ...] size
        src = batch[0].transpose(0, 1).to(device)
        trg_char = batch[1][0].transpose(0, 1).to(device)
        trg_word = batch[1][1].transpose(0, 1).to(device)
        optimizer.zero_grad()

        output = model(src, trg_char)
        # Convert prediction to words
        true = trg_word.transpose(0, 1).unsqueeze(1)
        true = [[[trg_vocab.itos[word_idx] for word_idx in comb] for comb in sent] for sent in true]
        pred = output.transpose(0, 1).argmax(-1)
        pred = [[trg_vocab.itos[word_idx] for word_idx in sent] for sent in pred]

        output = output[1:].reshape(-1, output.shape[-1])
        trg = trg_word[1:].reshape(-1)
        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        bleu += bleu_score(pred, true)

    return (epoch_loss / len(iterator)), (bleu / len(iterator))


def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module, trg_vocab: Vocab, device: torch.device):
    model.eval()

    epoch_loss = 0
    bleu = 0.0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(iterator), total=len(iterator)):
            # Need to convert to [seq_len x batch x ...] size
            src = batch[0].transpose(0, 1).to(device)
            trg_char = batch[1][0].transpose(0, 1).to(device)
            trg_word = batch[1][1].transpose(0, 1).to(device)

            output = model(src, trg_char, 0)  # turn off teacher forcing
            # Convert prediction to words
            true = trg_word.transpose(0, 1).unsqueeze(1)
            true = [[[trg_vocab.itos[word_idx] for word_idx in comb] for comb in sent] for sent in true]
            pred = output.transpose(0, 1).argmax(-1)
            pred = [[trg_vocab.itos[word_idx] for word_idx in sent] for sent in pred]

            output = output[1:].reshape(-1, output.shape[-1])
            trg = trg_word[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            bleu += bleu_score(pred, true)
    return (epoch_loss / len(iterator)), (bleu / len(iterator))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    train_dataset, val_dataset, test_dataset = get_dataset()
    train_iterator = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_iterator = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    input_dim = len(train_dataset.vocab[0])
    char_output_dim = len(train_dataset.vocab[1])
    word_output_dim = len(train_dataset.vocab[2])

    # enc_emb_dim = 256
    # dec_emb_dim = 256
    # enc_hid_dim = 512
    # dec_hid_dim = 512
    # attn_dim = 64
    # enc_dropout = 0.5
    # dec_dropout = 0.5

    enc_emb_dim = 300
    dec_emb_dim = 300
    enc_hid_dim = 64
    dec_hid_dim = 64
    attn_dim = 8
    enc_dropout = 0.5
    dec_dropout = 0.5

    enc_emb = WordCharCNNEmbedding(input_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout, enc_emb)
    attn = Attention(enc_hid_dim, dec_hid_dim, attn_dim)
    dec_emb = WordCharCNNEmbedding(char_output_dim)
    dec = Decoder(word_output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn, dec_emb)
    model = Seq2Seq(enc, dec, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab[2].stoi["<pad>"])
    optimizer = optim.Adam(model.parameters())
    print(f"The model has {count_parameters(model):,} trainable parameters")
    n_epochs = 10
    clip = 1

    best_valid_loss = float("inf")

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss, train_bleu_score = train(
            model, train_iterator, optimizer, criterion, clip, train_dataset.vocab[2], device
        )
        valid_loss, valid_bleu_score = evaluate(model, valid_iterator, criterion, train_dataset.vocab[2], device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} |  Train BLEU: {train_bleu_score:7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. BLEU: {valid_bleu_score:7.3f}"
        )

    test_loss = evaluate(model, test_iterator, criterion, train_dataset.vocab[2], device)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
