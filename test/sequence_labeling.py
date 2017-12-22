from torchtext import data
from torchtext import datasets

# Define the fields associated with the sequences.
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# Download and the load default data.
train, val, test = datasets.SequenceTaggingDataset.load_default_dataset(
    fields=(('word', WORD), ('udtag', UD_TAG), (None, None)))

print(train.fields)
print(len(train))
print(vars(train[0]))

# We can also define more than two columns.
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# Load the specified data.
train, val, test = datasets.SequenceTaggingDataset.splits(
    fields=(('word', WORD), ('udtag', UD_TAG), ('ptbtag', PTB_TAG)),
    path=".data/sequence-labeling/en-ud-v2",
    train="en-ud-tag.v2.train.txt",
    validation="en-ud-tag.v2.dev.txt",
    test="en-ud-tag.v2.test.txt")

print(train.fields)
print(len(train))
print(vars(train[0]))

WORD.build_vocab(train.word, min_freq=3)
UD_TAG.build_vocab(train.udtag)
PTB_TAG.build_vocab(train.ptbtag)

print(UD_TAG.vocab.freqs)
print(PTB_TAG.vocab.freqs)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=3, device=0)

batch = next(iter(train_iter))

print("words", batch.word)
print("udtags", batch.udtag)
print("ptbtags", batch.ptbtag)
