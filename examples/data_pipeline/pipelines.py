import torch
from transforms import PretrainedSPTokenizer, PretrainedSPVocab
from torchtext.experimental.functional import (
    sequential_transforms,
    totensor,
)


tokenizer = PretrainedSPTokenizer('m_user.model')
vocab = PretrainedSPVocab('m_user.model')

# Insert token in vocab to match a pretrained vocab
vocab.insert_token('<pad>', 1)
txt_transform1 = sequential_transforms(tokenizer, vocab, totensor(dtype=torch.long))
txt_transform1('here is an example')

