import torch
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
from torchtext.functional import to_tensor
from torchtext.transforms import LabelToIndex
from torchtext.experimental.datasets.sst2 import SST2
from torch.optim import Adam
import torch.nn as nn
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def get_model():
    xlmr_large = XLMR_BASE_ENCODER
    classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
    model = xlmr_large.get_model(head=classifier_head)
    return model


batch_size = 64
criteria = nn.CrossEntropyLoss()
model = get_model()
model.to(device)
optim = Adam(model.parameters())


def train_step(model, optim, input, target, criteria):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(model, input, target, criteria):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def get_processed_datapipe(batch_size, split='train'):
    text_transform = XLMR_BASE_ENCODER.transform()
    label_transform = LabelToIndex(label_names=['0', '1'])
    dp = SST2(split=split).batch(batch_size).rows2columnar(["text", "label"])
    dp = dp.map(lambda x: {'token_ids': to_tensor(text_transform(x['text']), padding_value=text_transform.pad_idx),
                           'target': torch.tensor(label_transform(x["label"]))})

    return dp


def evaluate():
    model.eval()
    eval_dp = get_processed_datapipe(batch_size, split='dev')
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
