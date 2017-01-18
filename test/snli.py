from torchtext import data
from torchtext import datasets

TEXT = data.Field()
LABEL = data.Field(sequential=False)

train, val, test = datasets.SNLI.splits(TEXT, LABEL)

print(train.fields)
print(len(train))
print(vars(train[0]))

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=3, device=0)

batch = next(iter(train_iter))
print(batch.premise)
print(batch.hypothesis)
print(batch.label)

train_iter, val_iter, test_iter = datasets.SNLI.iters(batch_size=4)

batch = next(iter(train_iter))
print(batch.premise)
print(batch.hypothesis)
print(batch.label)