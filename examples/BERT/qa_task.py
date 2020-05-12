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


def process_raw_data(data):
    _data = []
    for item in data:
        right_length = True
        for _idx in range(len(item['ans_pos'])):
            if item['ans_pos'][_idx][1] + item['question'].size(0) + 2 >= args.bptt:
                right_length = False
        if right_length:
            _data.append(item)
    return _data


def collate_batch(batch):
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


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_batch)
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
            loss = (criterion(start_pos, target_start_pos[0])
                    + criterion(end_pos, target_end_pos[0])) / 2
            total_loss += loss.item()
            start_pos = nn.functional.softmax(start_pos, dim=1).argmax(1)
            end_pos = nn.functional.softmax(end_pos, dim=1).argmax(1)
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


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_batch)
    train_loss_log.append(0.0)
    for idx, (seq_input, ans_pos, tok_type) in enumerate(dataloader):
        optimizer.zero_grad()
        start_pos, end_pos = model(seq_input, token_type_input=tok_type)
        target_start_pos, target_end_pos = ans_pos[0].to(device).split(1, dim=-1)
        target_start_pos = target_start_pos.squeeze(-1)
        target_end_pos = target_end_pos.squeeze(-1)
        loss = (criterion(start_pos, target_start_pos) + criterion(end_pos, target_end_pos)) / 2
        loss.backward()
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
    parser.add_argument('--lr', type=float, default=5.0,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=72, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='max. sequence length for context + question')
    parser.add_argument('--seed', type=int, default=21192391,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='qa_model.pt',
                        help='path to save the final bert model')
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str, default='ns_bert.pt',
                        help='path to save the pretrained bert')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

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
    cls_id = vocab.stoi['<cls>']
    train_dataset, dev_dataset = SQuAD1(vocab=vocab)
    train_dataset.data = process_raw_data(train_dataset.data)
    dev_dataset.data = process_raw_data(dev_dataset.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_bert = torch.load(args.bert_model)
    model = QuestionAnswerTask(pretrained_bert).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
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
        if best_f1 is None or val_f1 > best_f1:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_f1 = val_f1
        else:
            scheduler.step()

    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_loss, test_exact, test_f1 = evaluate(dev_dataset)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | exact {:8.3f}% | f1 {:8.3f}%'.format(
        test_loss, test_exact, test_f1))
    print('=' * 89)
    print_loss_log('qa_loss.txt', train_loss_log, val_loss_log, test_loss)
    with open(args.save, 'wb') as f:
        torch.save(model, f)
