import itertools
import re
from functools import partial

import torch
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.experimental.datasets.raw.translation import DATASETS
from torchtext.experimental.datasets.translation import TranslationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func
from torchtext.vocab import build_vocab_from_iterator


def build_word_vocab(data, transforms, index, init_token="<w>", eos_token="</w>"):
    tok_list = [[init_token], [eos_token]]
    return build_vocab_from_iterator(tok_list + list(map(lambda x: transforms(x[index]), data)))


def build_char_vocab(
    data, transforms, index, init_word_token="<w>", eos_word_token="</w>", init_sent_token="<s>", eos_sent_token="</s>",
):
    tok_list = [
        [init_word_token],
        [eos_word_token],
        [init_sent_token],
        [eos_sent_token],
    ]
    for line in data:
        tokens = list(itertools.chain.from_iterable(transforms(line[index])))
        tok_list.append(tokens)
    return build_vocab_from_iterator(tok_list)


def char_vocab_func(vocab):
    def func(tok_iter):
        return [[vocab[char] for char in word] for word in tok_iter]

    return func


def special_char_tokens_func(
    init_word_token="<w>", eos_word_token="</w>", init_sent_token="<s>", eos_sent_token="</s>",
):
    def func(tok_iter):
        result = [[init_word_token, init_sent_token, eos_word_token]]
        result += [[init_word_token] + word + [eos_word_token] for word in tok_iter]
        result += [[init_word_token, eos_sent_token, eos_word_token]]
        return result

    return func


def special_word_token_func(init_word_token="<w>", eos_word_token="</w>"):
    def func(tok_iter):
        return [init_word_token] + tok_iter + [eos_word_token]

    return func


def parallel_transforms(*transforms):
    def func(txt_input):
        result = []
        for transform in transforms:
            result.append(transform(txt_input))
        return tuple(result)

    return func


def get_dataset(dataset_name: str):
    # Get the raw dataset first. This will give us the text
    # version of the dataset
    train, test, val = DATASETS[dataset_name]()
    # Cache training data for vocabulary construction
    train_data = [line for line in train]
    val_data = [line for line in val]
    test_data = [line for line in test]
    # Setup word tokenizer
    src_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    tgt_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    # Setup char tokenizer

    def remove_extra_whitespace(line):
        return re.sub(" {2,}", " ", line)

    src_char_transform = sequential_transforms(remove_extra_whitespace, src_tokenizer, partial(map, list))
    tgt_char_transform = sequential_transforms(remove_extra_whitespace, tgt_tokenizer, partial(map, list))
    tgt_word_transform = sequential_transforms(remove_extra_whitespace, tgt_tokenizer)

    # Setup vocabularies (both words and chars)
    src_char_vocab = build_char_vocab(train_data, src_char_transform, index=0)
    tgt_char_vocab = build_char_vocab(train_data, tgt_char_transform, index=1)
    tgt_word_vocab = build_word_vocab(train, tgt_word_transform, 0)

    # Building the dataset with character level tokenization
    src_char_transform = sequential_transforms(
        src_char_transform, special_char_tokens_func(), char_vocab_func(src_char_vocab)
    )
    tgt_char_transform = sequential_transforms(
        tgt_char_transform, special_char_tokens_func(), char_vocab_func(tgt_char_vocab)
    )
    tgt_word_transform = sequential_transforms(
        tgt_word_transform, special_word_token_func(), vocab_func(tgt_word_vocab)
    )
    tgt_transform = parallel_transforms(tgt_char_transform, tgt_word_transform)
    train_dataset = TranslationDataset(
        train_data, (src_char_vocab, tgt_char_vocab, tgt_word_vocab), (src_char_transform, tgt_transform)
    )
    val_dataset = TranslationDataset(
        val_data, (src_char_vocab, tgt_char_vocab, tgt_word_vocab), (src_char_transform, tgt_transform)
    )
    test_dataset = TranslationDataset(
        test_data, (src_char_vocab, tgt_char_vocab, tgt_word_vocab), (src_char_transform, tgt_transform)
    )

    return train_dataset, val_dataset, test_dataset
