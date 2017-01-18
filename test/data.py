import torch
from torchtext import data


TEXT = data.Field()
LABELS = data.Field()

train, val, test = data.Dataset.splits(
    path='~/chainer-research/jmt-data/pos_wsj/pos_wsj', train='.train',
    validation='.dev', test='.test', format='tsv',
    fields=[('text', TEXT), ('labels', LABELS)])

print(train.fields)
print(len(train))
print(vars(train[0]))

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=3, sort_key=lambda x: len(x.text), device=0)

LABELS.build_vocab(train.labels)
TEXT.build_vocab(train.text)

print(TEXT.vocab.freqs.most_common(10))
print(LABELS.vocab.itos)

batch = next(iter(train_iter))
print(batch.text)
print(batch.labels)
