import torch
from transforms import (
    PretrainedSPTokenizer,
    PretrainedSPVocab,
    TextDataPipeline,
)


tokenizer = PretrainedSPTokenizer('m_user.model')
jit_tokenizer = torch.jit.script(tokenizer)
print('jit tokenizer success!')

vocab = PretrainedSPVocab('m_user.model')
jit_vocab = torch.jit.script(vocab)
print('jit vocab success!')

# Insert token in vocab to match a pretrained vocab
vocab.insert_token('<pad>', 1)
txt_pipe1 = TextDataPipeline(tokenizer, vocab)
print(txt_pipe1('here is an example'))
print('txt_pipe1 success!')

jit_txt_pipe1 = torch.jit.script(txt_pipe1)
print('jit txt_pipe1 success!')
