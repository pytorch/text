# Tokenizer

This example demonstrates how you can use torchtext's `GPT2BPETokenizer` in a C++ environment.

## Steps

### 1. Download necessary artifacts

First we download `gpt2_bpe_vocab.bpe` and `gpt2_bpe_encoder.json` artifacts, both of which are needed to construct the
`GPT2BPETokenizer` object.

```bash
curl -O https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe
curl -O https://download.pytorch.org/models/text/gpt2_bpe_encoder.json
```

### 2. Create tokenizer TorchScript file

Next we create our tokenizer object, and save it as a TorchScript object. We also print out the output of the tokenizer
on a sample sentence and verify that the output is the same before and after saving and re-loading the tokenizer. In the
next steps we will load and execute the tokenizer in our C++ application. The C++ code is found in
[`main.cpp`](./main.cpp).

```bash
tokenizer_file="tokenizer.pt"
python create_tokenizer.py --tokenizer-file "${tokenizer_file}"
```

### 3. Build the application

Please refer to [the top level README.md](../README.md)

### 4. Run the application

Now we run the C++ application `tokenizer`, with the TorchScript object we created in Step 2. The tokenizer is run with
the following sentence as input and we verify that the output is the same as that of Step 2.

In [the top level directory](../)

```bash
./build/tokenizer/tokenize "tokenizer/${tokenizer_file}"
```
