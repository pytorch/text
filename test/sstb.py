from torchtext import data
from torchtext import datasets

# Approach 1:
# set up fields
TEXT = data.Field()
LABEL = data.Field(sequential=False)

# make splits for data
train, val, test = datasets.SSTBDataset.splits(TEXT, LABEL, fine=True,
                                               neutral=False)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
TEXT.build_vocab(train)
LABEL.build_vocab(train) # do I need to build a vocab for LABEL?
# also, the label vocab has <pad> in it and one more artifact.

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=3, device=0)

# print batch information
batch = next(iter(train_iter))
print(batch.sentence)
print(batch.label)

# Approach 2:
train_iter, val_iter, test_iter = datasets.SSTBDataset.iter(batch_size=4)

# print batch information
batch = next(iter(train_iter))
print(batch.sentence)
print(batch.label)
