# torch-text

This repository will consist of:

- [text.data](#data) : Generic data loaders, abstractions, and iterators for text
- [text.datasets](#datasets) : Pre-built loaders for common NLP datasets
- (maybe) text.models : Model definitions and pre-trained models for popular NLP examples
(though the situation is not the same as vision, where people can download a pretrained ImageNet model and immediately
make it useful for other tasks -- it might make more sense to leave NLP models in the torch/examples repo)

# Data

The data module should provide the following:

- Ability to describe declaratively how to load a custom NLP dataset that's in a "normal" format:
```python
pos = data.Dataset(
    path='data/pos/pos_wsj_train.tsv', format='tsv',
    fields=[('text', data.Field(time_series=True),
            ('labels', data.Field(time_series=True)])

sentiment = data.Dataset(
    path='data/sentiment/train.json', format='json',
    fields=[{'sentence_tokenized': ('text', data.Field(time_series=True)),
            ('sentiment_gold': ('labels', data.Field(time_series=False))])
```
- Ability to define a preprocessing pipeline:
```python
src = data.Field(time_series=True, preprocess=my_custom_tokenizer)
trg = data.Field(time_series=True, preprocess=my_custom_tokenizer)
mt_train = data.ParallelCorpus(
    path='data/mt/wmt16-ende.train', exts=('.en', '.de'),
    fields=(('src', src), ('trg', trg))
```
- Batching, padding, and numericalizing (including building a vocabulary object):
```python
# continuing from above
mt_dev = data.ParallelCorpus(
    path='data/mt/newstest2014', exts=('.en', '.de'),
    fields=(('src', src), ('trg', trg))
src.build_vocab(mt_train.src, max_size=80000)
trg.build_vocab(mt_train.trg, max_size=40000)
# mt_dev shares the fields, so it shares their vocab objects

train_iter = data.BucketIterator(
    batch_size=32, mt_train,
    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
# usage
>>>next(train_iter)
<data.Batch(batch_size=32, src=[LongTensor (32, 25)], trg=[LongTensor (32, 28)])>
```

# Datasets

Some datasets it would be useful to have built in:

- bAbI and successors from FAIR
- SST and IMDb sentiment
- SNLI
- Penn Treebank (for language modeling and parsing)
- WMT and/or IWSLT machine translation
- SQuAD
