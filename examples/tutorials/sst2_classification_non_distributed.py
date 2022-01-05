"""
SST-2 Binary text classification with XLM-RoBERTa model
=======================================================

**Author**: `Parmeet Bhatia <parmeetbhatia@fb.com>`__

"""

######################################################################
# Overview
# --------
#
# This tutorial shows how to train a text classifier using pre-trained XLM-RoBERTa model.
# TODO


######################################################################
# Common imports
# --------------

import torch
import torch.nn as nn
DEVICE = torch.DEVICE("cuda") if torch.cuda.is_available() else "cpu"


#######################################################################
# Data Transformation
# -------------------
# TODO

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

PADDING_IDX = 1
BOS_IDX = 0
EOS_IDX = 2
MAX_SEQ_LEN = 512
XLMR_VOCAB_PATH = r"https:/download.pytorch.org/models/text/xlmr.vocab.pt"
XLMR_SPM_MODEL_PATH = r"https:/download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

TEXT_TRANSFORM = nn.Sequential(
    T.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH),
    T.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH)),
    T.Truncate(MAX_SEQ_LEN - 2),
    T.AddToken(token=BOS_IDX, begin=True),
    T.AddToken(token=EOS_IDX, begin=False),
)

# Alternately we can also use transform shipped with pre-trained model that does all of the above out-of-the-box
# TEXT_TRANSFORM = XLMR_BASE_ENCODER.transform()

LABEL_TRANSFORM = T.LabelToIndex(label_names=['0', '1'])

#######################################################################
# Dataset
# -------
# TODO

from torchtext.experimental.datasets.sst2 import SST2
BATCH_SIZE = 16

TRAIN_DATAPIPE = SST2(split='train')
DEV_DATAPIPE = SST2(split='dev')

# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
TRAIN_DATAPIPE = TRAIN_DATAPIPE.map(lambda x: (TEXT_TRANSFORM(x[0]), LABEL_TRANSFORM(x[1])))
TRAIN_DATAPIPE = TRAIN_DATAPIPE.batch(BATCH_SIZE)
TRAIN_DATAPIPE = TRAIN_DATAPIPE.rows2columnar(["token_ids", "target"])

DEV_DATAPIPE = DEV_DATAPIPE.map(lambda x: (TEXT_TRANSFORM(x[0]), LABEL_TRANSFORM(x[1])))
DEV_DATAPIPE = DEV_DATAPIPE.batch(BATCH_SIZE)
DEV_DATAPIPE = DEV_DATAPIPE.rows2columnar(["token_ids", "target"])

# Alternately we can also use batched API (i.e apply transformation on the whole batch)
# TRAIN_DATAPIPE = TRAIN_DATAPIPE.batch(BATCH_SIZE).rows2columnar(["text", "label"])
# TRAIN_DATAPIPE = TRAIN_DATAPIPE.map(lambda x: {"token_ids": TEXT_TRANSFORM(x["text"]), "target": LABEL_TRANSFORM(x["label"])})
# DEV_DATAPIPE = DEV_DATAPIPE.batch(BATCH_SIZE).rows2columnar(["text", "label"])
# DEV_DATAPIPE = DEV_DATAPIPE.map(lambda x: {"token_ids": TEXT_TRANSFORM(x["text"]), "target": LABEL_TRANSFORM(x["label"])})

######################################################################
# Model Preparation
# -----------------
# TODO
NUM_CLASSES = 2
INPUT_DIM = 768

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
classifier_head = RobertaClassificationHead(num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
MODEL = XLMR_BASE_ENCODER.get_model(head=classifier_head)
MODEL.to(DEVICE)


#######################################################################
# Training methods
# ----------------
# TODO

import torchtext.functional as F
from torch.optim import AdamW

LEARNING_RATE = 1e-5
OPTIM = AdamW(MODEL.parameters(), lr=LEARNING_RATE)
CRITERIA = nn.CrossEntropyLoss()


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
        for batch in DEV_DATAPIPE:
            input = F.to_tensor(batch['token_ids'], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


#######################################################################
# Train
# -----
# TODO

NUM_EPOCHS = 1

for e in range(NUM_EPOCHS):
    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
    MODEL.train()
    for batch in TRAIN_DATAPIPE:
        input = F.to_tensor(batch['token_ids'], padding_value=PADDING_IDX).to(DEVICE)
        target = torch.tensor(batch['target']).to(DEVICE)
        train_step(input, target)

loss, accuracy = evaluate()
print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(NUM_EPOCHS), loss, accuracy)
