"""
SST-2 Binary text classification with XLM-RoBERTa model
=======================================================

**Author**: `Parmeet Bhatia <parmeetbhatia@fb.com>`__

"""

######################################################################
# Overview
# --------
#
# This tutorial demonstrates how to train a text classifier on SST-2 binary dataset using a pre-trained XLM-RoBERTa (XLM-R) model.
# We will show how to use torchtext library to:
#
# 1. build text pre-processing pipeline for XLM-R model
# 2. read SST-2 dataset and transform it using text and label transformation
# 3. instantiate classification model using pre-trained XLM-R encoder
#
#
# To run this tutorial, please install torchtext nightly and TorchData (following commands will do in google Colab)
#
# ::
#
#   !pip3 install --pre --upgrade torchtext -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
#   !pip install --user "git+https://github.com/pytorch/data.git"
#


######################################################################
# Common imports
# --------------
import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#######################################################################
# Data Transformation
# -------------------
#
# Models like XLM-R cannot work directly with raw text. The first step in training
# these models is to transform input text into tensor (numerical) form such that it
# can then be processed by models to make predictions. A standard way to process text is:
#
# 1. Tokenize text
# 2. Convert tokens into (integer) IDs
# 3. Add any special tokens IDs
#
# XLM-R uses sentencepiece model for text tokenization. Below, we use pre-trained sentencepiepce
# model along with corresponding vocabulary to build text pre-processing pipeline using torchtext's transforms.
# The transforms are pipelined using :py:func:`torchtext.transforms.Sequential` which is similar to :py:func:`torch.nn.Sequential`
# but is torchscriptable. Note that the transforms support both batched and non-batched text inputs i.e, one
# can either pass a single sentence or list of sentences.
#

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)


from torch.utils.data import DataLoader

#######################################################################
# Alternately we can also use transform shipped with pre-trained model that does all of the above out-of-the-box
#
# ::
#
#   text_transform = XLMR_BASE_ENCODER.transform()
#

#######################################################################
# Dataset
# -------
# torchtext provides several standard NLP datasets. For complete list, refer to documentation
# at https://pytorch.org/text/stable/datasets.html. These datasets are build using composable torchdata
# datapipes and hence support standard flow-control and mapping/transformation using user defined functions
# and transforms. Below, we demonstrate how to use text and label processing transforms to pre-process the
# SST-2 dataset.
#
#

from torchtext.datasets import SST2

batch_size = 16

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")

# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)


#######################################################################
# Alternately we can also use batched API (i.e apply transformation on the whole batch)
#
# ::
#
#   train_datapipe = train_datapipe.batch(batch_size).rows2columnar(["text", "label"])
#   train_datapipe = train_datapipe.map(lambda x: {"token_ids": text_transform(x["text"]), "target": label_transform(x["label"])})
#   dev_datapipe = dev_datapipe.batch(batch_size).rows2columnar(["text", "label"])
#   dev_datapipe = dev_datapipe.map(lambda x: {"token_ids": text_transform(x["text"]), "target": label_transform(x["label"])})
#

######################################################################
# Model Preparation
# -----------------
#
# torchtext provides SOTA pre-trained models that can be used to fine-tune on downstream NLP tasks.
# Below we use pre-trained XLM-R encoder with standard base architecture and attach a classifier head to fine-tune it
# on SST-2 binary classification task. We shall use standard Classifier head from the library, but users can define
# their own appropriate task head and attach it to the pre-trained encoder. For additional details on available pre-trained models,
# please refer to documentation at https://pytorch.org/text/main/models.html
#
#

num_classes = 2
input_dim = 768

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)


#######################################################################
# Training methods
# ----------------
#
# Let's now define the standard optimizer and training criteria as well as some helper functions
# for training and evaluation
#

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
        for batch in dev_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


#######################################################################
# Train
# -----
#
# Now we have all the ingredients to train our classification model. Note that we are able to directly iterate
# on our dataset object without using DataLoader. Our pre-process dataset  shall yield batches of data already,
# thanks to the batching datapipe we have applied. For distributed training, we would need to use DataLoader to
# take care of data-sharding.
#

num_epochs = 1

for e in range(num_epochs):
    for batch in train_dataloader:
        input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch["target"]).to(DEVICE)
        train_step(input, target)

    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))


#######################################################################
# Output
# ------
#
# ::
#
#   100%|██████████|5.07M/5.07M [00:00<00:00, 40.8MB/s]
#   Downloading: "https://download.pytorch.org/models/text/xlmr.vocab.pt" to /root/.cache/torch/hub/checkpoints/xlmr.vocab.pt
#   100%|██████████|4.85M/4.85M [00:00<00:00, 16.8MB/s]
#   Downloading: "https://download.pytorch.org/models/text/xlmr.base.encoder.pt" to /root/.cache/torch/hub/checkpoints/xlmr.base.encoder.pt
#   100%|██████████|1.03G/1.03G [00:26<00:00, 47.1MB/s]
#   Epoch = [0], loss = [0.2629831412637776], accuracy = [0.9105504587155964]
#
