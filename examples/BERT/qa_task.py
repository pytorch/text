import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from data import SQuAD1
from model import QuestionAnswerTask
from metrics import compute_qa_exact, compute_qa_f1
from utils import print_loss_log


def pad_squad_data(batch):
    # Fix sequence length to args.bptt with padding or trim
    seq_list = []
    ans_pos_list = []
    tok_type = []

    for item in batch:
        qa_item = torch.cat((torch.tensor([cls_id]), item['question'], torch.tensor([sep_id]),
                             item['context'], torch.tensor([sep_id])))
        if qa_item.size(0) > args.bptt:
            qa_item = qa_item[:args.bptt]
        elif qa_item.size(0) < args.bptt:
            qa_item = torch.cat((qa_item,
                                 torch.tensor([pad_id] * (args.bptt -
                                              qa_item.size(0)))))
        seq_list.append(qa_item)
        pos_list = [pos + item['question'].size(0) + 2 for pos in item['ans_pos']]  # 1 for sep and 1 for cls
        ans_pos_list.append(pos_list)
        tok_type.append(torch.cat((torch.zeros((item['question'].size(0) + 2)),
                                   torch.ones((args.bptt -
                                               item['question'].size(0) - 2)))))

    _ans_pos_list = []
    for pos in zip(*ans_pos_list):
        _ans_pos_list.append(torch.stack(list(pos)))
    return torch.stack(seq_list).long().t().contiguous().to(device), \
        _ans_pos_list, \
        torch.stack(tok_type).long().t().contiguous().to(device)

###############################################################################
# Evaluating code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            collate_fn=pad_squad_data)
    ans_pred_tokens_samples = []
    vocab = data_source.vocab

    with torch.no_grad():
        for idx, (seq_input, ans_pos_list, tok_type) in enumerate(dataloader):
            start_pos, end_pos = model(seq_input, token_type_input=tok_type)
            target_start_pos, target_end_pos = [], []
            for item in ans_pos_list:
                _target_start_pos, _target_end_pos = item.to(device).split(1, dim=-1)
                target_start_pos.append(_target_start_pos.squeeze(-1))
                target_end_pos.append(_target_end_pos.squeeze(-1))

            # in dev, pos come with three set. Use the first one to calculate loss here
            loss = (criterion(start_pos, target_start_pos[0])
                    + criterion(end_pos, target_end_pos[0])) / 2
            total_loss += loss.item()

            start_pos = nn.functional.softmax(start_pos, dim=1).argmax(1)
            end_pos = nn.functional.softmax(end_pos, dim=1).argmax(1)

            # [TODO] remove '<unk>', '<cls>', '<pad>', '<MASK>' from ans_tokens and pred_tokens
            # Go through batch and convert ids to tokens list
            seq_input = seq_input.transpose(0, 1)  # convert from (S, N) to (N, S)
            for num in range(0, seq_input.size(0)):
                if int(start_pos[num]) > int(end_pos[num]):
                    continue  # start pos is in front of end pos
                ans_tokens = []
                for _idx in range(len(target_end_pos)):
                    ans_tokens.append([vocab.itos[int(seq_input[num][i])]
                                       for i in range(target_start_pos[_idx][num],
                                                      target_end_pos[_idx][num] + 1)])
                pred_tokens = [vocab.itos[int(seq_input[num][i])]
                               for i in range(start_pos[num],
                                              end_pos[num] + 1)]
                ans_pred_tokens_samples.append((ans_tokens, pred_tokens))

    return total_loss / (len(data_source) // batch_size), \
        compute_qa_exact(ans_pred_tokens_samples), \
        compute_qa_f1(ans_pred_tokens_samples)


###############################################################################
# Training code
###############################################################################

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=pad_squad_data)
    train_loss_log.append(0.0)

    for idx, (seq_input, ans_pos, tok_type) in enumerate(dataloader):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        start_pos, end_pos = model(seq_input, token_type_input=tok_type)

        target_start_pos, target_end_pos = ans_pos[0].to(device).split(1, dim=-1)
        target_start_pos = target_start_pos.squeeze(-1)
        target_end_pos = target_end_pos.squeeze(-1)
        loss = (criterion(start_pos, target_start_pos) + criterion(end_pos, target_end_pos)) / 2
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            train_loss_log[-1] = cur_loss
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, idx,
                                                      len(train_dataset) // batch_size,
                                                      scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / args.log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question-Answer fine-tuning task')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--lr', type=float, default=5,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                        help='batch size')
    # [TODO] increase bptt to 200
    parser.add_argument('--bptt', type=int, default=35,
                        help='max. sequence length for context + question')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='qa_model.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str,
                        help='path to save the pretrained bert')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################
    # Load data
    ###################################################################

    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, dev_dataset = SQuAD1()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)
    pad_id = vocab.stoi['<pad>']
    sep_id = vocab.stoi['<sep>']
    # [DONE] add cls_id and attach to the beginning of sequence
    cls_id = vocab.stoi['<cls>']

    # [TODO] switch to SQuAD 2.0
    train_dataset, dev_dataset = SQuAD1(vocab=vocab)

    # Remove data with 'question' + 'context' > args.bptt or
    #[TODO] remove the cases with pos larger than args.bptt
    def clean_data(data):
        _data = []
        for item in data:
            right_length = True
            for _idx in range(len(item['ans_pos'])):
                #[TODO] remove the cases with pos larger than args.bptt
                if item['ans_pos'][_idx][1] + item['question'].size(0) + 2 >= args.bptt: # 2 for '<cls>' '<sep>'
                    right_length = False
            if right_length:
                _data.append(item)
        return _data
    train_dataset.data = clean_data(train_dataset.data)
    dev_dataset.data = clean_data(dev_dataset.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################
    ###################################################################
    # Build the model
    ###################################################################

    pretrained_bert = torch.load(args.bert_model)
    model = QuestionAnswerTask(pretrained_bert).to(device)

    criterion = nn.CrossEntropyLoss()

    ###################################################################
    # Loop over epochs.
    ###################################################################

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
#    best_val_loss = None
    best_f1 = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, val_exact, val_f1 = evaluate(dev_dataset)
        val_loss_log.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'exact {:8.3f}% | '
              'f1 {:8.3f}%'.format(epoch, (time.time() - epoch_start_time),
                                   val_loss, val_exact, val_f1))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
#        if not best_val_loss or val_loss < best_val_loss:
        if best_f1 is None or val_f1 > best_f1:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
#            best_val_loss = val_loss
            best_f1 = val_f1
        else:
            scheduler.step()

    ###################################################################
    # Load the best saved model.
    ###################################################################
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    ###################################################################
    # Run on test data.
    ###################################################################
    test_loss, test_exact, test_f1 = evaluate(dev_dataset)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | exact {:8.3f}% | f1 {:8.3f}%'.format(
        test_loss, test_exact, test_f1))
    print('=' * 89)
    print_loss_log('qa_loss.txt', train_loss_log, val_loss_log, test_loss)

    with open('fine_tuning_qa_model.pt', 'wb') as f:
        torch.save(model, f)
#python qa_task.py --bert-model squad_vocab_pretrained_bert.pt --epochs 2 --save-vocab squad_vocab.pt
