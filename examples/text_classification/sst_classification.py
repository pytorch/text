from typing import OrderedDict
import torch
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
import torchtext.transforms as transforms
from torchtext.experimental.datasets.sst2 import SST2
from torch.hub import load_state_dict_from_url
from torch.optim import Adam
import torch.nn as nn
from collections import OrderedDict
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
MODEL = XLMR_BASE_ENCODER.get_model(head=classifier_head)
MODEL.to(device)
batch_size = 64
CRITERIA = nn.CrossEntropyLoss()
OPTIM = Adam(MODEL.parameters())
PADDING_IDX = 1
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
TEXT_TRANSFORM = XLMR_BASE_ENCODER.transform()
LABEL_TRANSFORM = transforms.LabelToIndex(label_names=['0', '1'])


XLMR_VOCAB_PATH = "https://download.pytorch.org/models/text/xlmr.vocab.pt",
XLMR_SPM_MODEL_PATH = "https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
TEXT_TRANSFORM = nn.Sequential(OrderedDict([
    ('tokenize', transforms.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH)),
    ('add_bos', transforms.AddToken(BOS_TOKEN, begin=True)),
    ('add_eos', transforms.AddToken(EOS_TOKEN, begin=False)),
    ('vocab', transforms.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH))),
])
)


TRAIN_DATAPIPE = SST2(split='train')
TRAIN_DATAPIPE.map(lambda x: (TEXT_TRANSFORM(x[0]), LABEL_TRANSFORM(x[1])))

DEV_DATAPIPE = SST2(split='dev')
DEV_DATAPIPE.map(lambda x: (TEXT_TRANSFORM(x[0]), LABEL_TRANSFORM(x[1])))

# Alternately we can also use batch API
# TRAIN_DATAPIPE = TRAIN_DATAPIPE.batch(batch_size).rows2columnar(["text", "label"])
# TRAIN_DATAPIPE.map(lambda x:{"token_ds":TEXT_TRANSFORM(x["text"]), "target":LABEL_TRANSFORM(x["label"])})


def train_step(input, target):
    output = MODEL(input)
    loss = CRITERIA(output, target)
    OPTIM.zero_grad()
    loss.backward()
    OPTIM.step()


def eval_step(input, target):
    output = MODEL(input)
    loss = CRITERIA(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    MODEL.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dp):
            input = batch['token_ids'].to(device)
            target = batch['target'].to(device)
            loss, predictions = eval_step(model, input, target, criteria)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def train(num_epochs):
    train_dp = get_processed_datapipe(batch_size)
    for e in range(num_epochs):
        loss, accuracy = evaluate()
        print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
        model.train()
        for i, batch in enumerate(train_dp):
            input = batch['token_ids'].to(device)
            target = batch['target'].to(device)
            train_step(model, optim, input, target, criteria)

    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(num_epochs, loss, accuracy))
