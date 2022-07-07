# Tokenizer

This example demonstrates how you can use torchtext's `GPT2BPETokenizer` in a C++ environment.

## Steps

### 1. Create augmentation pipeline TorchScript file.

First we create our tokenizer object, and save it as a TorchScript object. We also print out the output of the tokenizer
on a sample sentence and verify that the output is the same before and after saving and re-loading the tokenizer. In the
next steps we will load and execute the tokenizer in our C++ application. The C++ code is found in
[`main.cpp`](./main.cpp).

```python
tokenizer_file="tokenizer.pt"
python create_tokenizer.py --tokenizer-file "${tokenizer_file}"
```

### 2. Build the application

Please refer to [the top level README.md](../README.md)

### 3. Run the application

Now we run the C++ application `tokenizer`, with the TorchScript object we created in Step 1. The tokenizer is run with
the following sentence as input and we verify that the output is the same as that of Step 1.

In [the top level directory](../)

```bash
./build/tokenizer/tokenize "tokenizer/${tokenizer_file}"
```
