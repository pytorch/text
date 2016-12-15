import torch
from text.torchtext import data


TEXT = data.Field(time_series=True)
LABELS = data.Field(time_series=True)

train = data.Dataset(path='~/data/pos_wsj/pos_wsj.train',
                     format='tsv', fields=[('text', TEXT), ('labels', LABELS)])

print(train.fields)
print(vars(train[0]))

train_iter = data.BucketIterator(
    train, batch_size=3, key=lambda x: len(x.text), device=0)

LABELS.build_vocab(train.labels)
TEXT.build_vocab(train.text)

print(TEXT.vocab.freqs.most_common(10))
print(LABELS.vocab.itos)

batch = next(iter(train_iter))
print(batch.text)
print(batch.labels)
