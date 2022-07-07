# Libtorchtext Examples

- [Tokenizer](./tokenizer)

## Build

The example applications in this directory depend on `libtorch` and `libtorchtext`. If you have a working `PyTorch`, you
already have `libtorch`. Please refer to
[this tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) for the use of `libtorch` and
TorchScript.

`libtorchtext` is the library of torchtext's C++ components without Python components. It is currently not distributed,
and it will be built alongside with the applications.

To build `libtorchtext` and the example applications you can run the following command.

```bash
chmod +x build.sh # give script execute permission
./build.sh
```

For the usages of each application, refer to the corresponding application directory.
