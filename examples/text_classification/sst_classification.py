import torch
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
import torchtext.transforms as T
import torchtext.functional as F
from torchtext.experimental.datasets.sst2 import SST2
from torch.hub import load_state_dict_from_url
from torch.optim import Adam
import torch.nn as nn
from collections import OrderedDict
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
MODEL = XLMR_BASE_ENCODER.get_model(head=classifier_head)
MODEL.to(device)
BATCH_SIZE = 16
CRITERIA = nn.CrossEntropyLoss()
OPTIM = Adam(MODEL.parameters())
PADDING_IDX = 1
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
MAX_SEQ_LEN = 512
NUM_EPOCHS = 1

XLMR_VOCAB_PATH = "https://download.pytorch.org/models/text/xlmr.vocab.pt",
XLMR_SPM_MODEL_PATH = "https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"


# Support both Non-Batched (single sentence) and Batched (List of sentences) as inputs
TEXT_TRANSFORM = nn.Sequential(OrderedDict([
    ('tokenize', T.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH)),  # tokenize using pre-trained SPM
    ('truncate', T.Truncate(MAX_SEQ_LEN - 2)),  # Truncate sequence to max allowable length
    ('add_bos', T.AddToken(BOS_TOKEN, begin=True)),  # Add BOS token at start of sequence
    ('add_eos', T.AddToken(EOS_TOKEN, begin=False)),  # Add EOS token at end of sequence
    ('vocab', T.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH))),  # Convert tokens into IDs
])
)

# Alternately we can also use transform shipped with pre-trained model
# TEXT_TRANSFORM = XLMR_BASE_ENCODER.transform()

LABEL_TRANSFORM = T.LabelToIndex(label_names=['0', '1'])

# using non-batched API
TRAIN_DATAPIPE = SST2(split='train')
TRAIN_DATAPIPE.map(lambda x: (TEXT_TRANSFORM(x[0]), LABEL_TRANSFORM(x[1])))
TRAIN_DATAPIPE = TRAIN_DATAPIPE.batch(BATCH_SIZE)
TRAIN_DATAPIPE = TRAIN_DATAPIPE.rows2columnar(["token_ids", "target"])

DEV_DATAPIPE = SST2(split='dev')
DEV_DATAPIPE.map(lambda x: (TEXT_TRANSFORM(x[0]), LABEL_TRANSFORM(x[1])))
DEV_DATAPIPE = DEV_DATAPIPE.batch(BATCH_SIZE)
DEV_DATAPIPE = DEV_DATAPIPE.rows2columnar(["token_ids", "target"])

# Alternately we can also use batched API
# TRAIN_DATAPIPE = TRAIN_DATAPIPE.batch(BATCH_SIZE).rows2columnar(["text", "label"])
# TRAIN_DATAPIPE.map(lambda x:{"token_ds":TEXT_TRANSFORM(x["text"]), "target":LABEL_TRANSFORM(x["label"])})
# DEV_DATAPIPE = DEV_DATAPIPE.batch(BATCH_SIZE).rows2columnar(["text", "label"])
# DEV_DATAPIPE.map(lambda x:{"token_ds":TEXT_TRANSFORM(x["text"]), "target":LABEL_TRANSFORM(x["label"])})


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
        for i, batch in enumerate(DEV_DATAPIPE):
            input = F.to_tensor(batch['token_ids'], padding_value=PADDING_IDX).to(device)
            target = torch.tensor(batch['target'].to(device))
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def train():
    for e in range(NUM_EPOCHS):
        loss, accuracy = evaluate()
        print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
        MODEL.train()
        for i, batch in enumerate(TRAIN_DATAPIPE):
            input = batch['token_ids'].to(device)
            target = batch['target'].to(device)
            train_step(input, target)

    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(NUM_EPOCHS), loss, accuracy)


if __name__ == "__main__":
    train()
