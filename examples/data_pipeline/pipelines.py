import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    TextDataPipeline,
)
from torchtext.experimental.transforms import (
    BasicEnglishNormalize,
)
from torchtext.experimental.vocab import vocab_from_file_object
from torchtext.experimental.vectors import FastText


tokenizer = PretrainedSPTokenizer('m_user.model')
jit_tokenizer = torch.jit.script(tokenizer)
print('jit SPM tokenizer success!')

vocab = PretrainedSPVocab('m_user.model')
jit_vocab = torch.jit.script(vocab)
print('jit SPM vocab success!')

# Insert token in vocab to match a pretrained vocab
vocab.insert_token('<pad>', 1)
txt_pipe1 = TextDataPipeline(tokenizer, vocab)
print(txt_pipe1('here is an example'))
print('txt_pipe1 success!')

jit_txt_pipe1 = torch.jit.script(txt_pipe1)
print('jit txt_pipe1 success!')

tokenizer = BasicEnglishNormalize()
jit_tokenizer = torch.jit.script(tokenizer)
print('jit Basic English tokenizer success!')

f = open('vocab.txt', 'r')
vocab = vocab_from_file_object(f)
jit_vocab = torch.jit.script(vocab)
print('jit HF vocab success!')

# Insert token in vocab to match a pretrained vocab
pipeline2 = TextDataPipeline(tokenizer, vocab.lookup_indices)
print(pipeline2('here is an example'))
print('pipeline2 success!')

f = open('vocab.txt', 'r')
vector = FastText()
jit_vocab = torch.jit.script(vector)
print('jit FastText vector success!')

# Insert token in vocab to match a pretrained vocab
pipeline3 = TextDataPipeline(tokenizer, vector.lookup_vectors)
print(pipeline3('here is an example'))
print('pipeline3 success!')
