import sys, os

import torcharrow as ta
import torchtext.transforms as T
from benchmark.utils import Timer
from torcharrow import functional as ta_F
from torchtext._download_hooks import load_state_dict_from_url
from torchtext.datasets import SST2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../examples"))
from data_pipeline.roberta_dataframe import init_ta_gpt2bpe_encoder, init_ta_gpt2bpe_vocab


def run_torchtext_ops():
    # tokenizer converting text into tokens
    encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
    tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)

    # vocabulary converting tokens to IDs
    vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
    vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))

    # add token to beginning or end of sentence
    add_bos_str = T.AddToken(token="<bos>", begin=True)
    add_eos_str = T.AddToken(token="<eros>", begin=False)
    add_bos_int = T.AddToken(token=0, begin=True)
    add_eos_int = T.AddToken(token=-1, begin=False)

    # dataset
    train_dp = SST2(split="train")
    text_list = list(train_dp.map(lambda x: x[0]))

    with Timer("Running torchtext's GPT2BPE tokenizer"):
        tokenized_text = tokenizer(text_list)

    with Timer("Running torchtext's vocab query"):
        token_ids = vocab(tokenized_text)

    with Timer("Running torchtext's add tokens operation (string)"):
        add_bos_str(token_ids)
        add_eos_str(token_ids)

    with Timer("Running torchtext's add tokens operation (int)"):
        add_bos_int(token_ids)
        add_eos_int(token_ids)


def run_torcharrow_ops():
    # tokenizer converting text into tokens
    tokenizer = init_ta_gpt2bpe_encoder()

    # vocabulary converting tokens to IDs
    vocab = init_ta_gpt2bpe_vocab()

    # dataset
    train_dp = SST2(split="train")
    text_list = list(train_dp.map(lambda x: x[0]))
    data_frame = ta.dataframe({"text": text_list})

    with Timer("Running torcharrow's GPT2BPE tokenizer"):
        data_frame["tokens"] = ta_F.bpe_tokenize(tokenizer, data_frame["text"])

    with Timer("Running torcharrow's vocab query"):
        data_frame["token_ids"] = ta_F.lookup_indices(vocab, data_frame["tokens"])

    with Timer("Running torcharrow's add tokens operation (string)"):
        ta_F.add_tokens(data_frame["tokens"], ["<bos>"], begin=True)
        ta_F.add_tokens(data_frame["tokens"], ["<eos>"], begin=False)

    with Timer("Running torcharrow's add tokens operation (int)"):
        ta_F.add_tokens(data_frame["token_ids"], [0], begin=True)
        ta_F.add_tokens(data_frame["token_ids"], [-1], begin=False)


if __name__ == "__main__":
    run_torchtext_ops()
    run_torcharrow_ops()
