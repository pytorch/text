import torch
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
import torchtext.transforms as T
import torchtext.functional as F
from torchtext.experimental.datasets.sst2 import SST2
from torch.hub import load_state_dict_from_url
from torch.optim import AdamW
import torch.nn as nn
DEVICE = torch.DEVICE("cuda") if torch.cuda.is_available() else "cpu"


classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
MODEL = XLMR_BASE_ENCODER.get_model(head=classifier_head)
MODEL.to(DEVICE)

# Model specific variables
PADDING_IDX = 1  # padding index
BOS_TOKEN = '<s>'  # begin of sentence token
EOS_TOKEN = '</s>'  # end of sentence token
MAX_SEQ_LEN = 512  # maximum length for the input sequence
XLMR_VOCAB_PATH = r"https:/download.pytorch.org/models/text/xlmr.vocab.pt"  # pre-trained vocabulary correspond to sentencepiece model for XLM-Roberta model
XLMR_SPM_MODEL_PATH = r"https:/download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"  # pre-trained sentencepeice model for XLM-Roberta model

# Training specific variables
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
CRITERIA = nn.CrossEntropyLoss()
OPTIM = AdamW(MODEL.parameters(), lr=LEARNING_RATE)


# Support both Non-Batched (single sentence) and Batched (List of sentences) as inputs
TEXT_TRANSFORM = nn.Sequential(
    T.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH),  # tokenize using pre-trained SPM
    T.Truncate(MAX_SEQ_LEN - 2),  # Truncate sequence to max allowable length
    T.AddToken(BOS_TOKEN, begin=True),  # Add BOS token at start of sequence
    T.AddToken(EOS_TOKEN, begin=False),  # Add EOS token at end of sequence
    T.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH)),  # Convert tokens into IDs
)

# Alternately we can also use transform shipped with pre-trained model that does all of the above out-of-the-box
# TEXT_TRANSFORM = XLMR_BASE_ENCODER.transform()

LABEL_TRANSFORM = T.LabelToIndex(label_names=['0', '1'])

# get the sst-2 dataset for 'train' and 'dev' split
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
        # we can directly iterate on DataPipe dataset or create DataLoader
        # for distributed and multi-processing we would need to use DataLoader
        for batch in DEV_DATAPIPE:
            input = F.to_tensor(batch['token_ids'], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def train():
    for e in range(NUM_EPOCHS):
        # loss, accuracy = evaluate()
        # print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
        MODEL.train()
        # we can directly iterate on DataPipe dataset or create DataLoader
        # for distributed and multi-processing we would need to use DataLoader
        for batch in TRAIN_DATAPIPE:
            input = F.to_tensor(batch['token_ids'], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            train_step(input, target)

    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(NUM_EPOCHS), loss, accuracy)


if __name__ == "__main__":
    train()
