from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe

# Approach 1:
# set up fields
TEXT = data.Field(lower=True, batch_first=True)

# make splits for data
train, valid, test = datasets.WikiText2.splits(TEXT)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0])['text'][0:10])

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))

# make iterator for splits
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=3, bptt_len=30, device="cuda:0")

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.target)

# Approach 2:
train_iter, valid_iter, test_iter = datasets.WikiText2.iters(batch_size=4, bptt_len=30)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.target)
