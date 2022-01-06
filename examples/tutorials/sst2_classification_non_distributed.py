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
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#######################################################################
# Data Transformation
# -------------------
# TODO

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 512
xlmr_vocab_path = r"https:/download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https:/download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = nn.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

# Alternately we can also use transform shipped with pre-trained model that does all of the above out-of-the-box
# text_transform = XLMR_BASE_ENCODER.transform()

label_transform = T.LabelToIndex(label_names=['0', '1'])

#######################################################################
# Dataset
# -------
# TODO

from torchtext.experimental.datasets.sst2 import SST2
batch_size = 16

train_datapipe = SST2(split='train')
dev_datapipe = SST2(split='dev')

# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), label_transform(x[1])))
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])

dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), label_transform(x[1])))
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])

# Alternately we can also use batched API (i.e apply transformation on the whole batch)
# train_datapipe = train_datapipe.batch(batch_size).rows2columnar(["text", "label"])
# train_datapipe = train_datapipe.map(lambda x: {"token_ids": text_transform(x["text"]), "target": label_transform(x["label"])})
# dev_datapipe = dev_datapipe.batch(batch_size).rows2columnar(["text", "label"])
# dev_datapipe = dev_datapipe.map(lambda x: {"token_ids": text_transform(x["text"]), "target": label_transform(x["label"])})

######################################################################
# Model Preparation
# -----------------
# TODO
num_classes = 2
input_dim = 768

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)


#######################################################################
# Training methods
# ----------------
# TODO

import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()


def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_datapipe:
            input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
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

num_epochs = 1

for e in range(num_epochs):
    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
    model.train()
    for batch in train_datapipe:
        input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch['target']).to(DEVICE)
        train_step(input, target)

loss, accuracy = evaluate()
print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(num_epochs), loss, accuracy)
