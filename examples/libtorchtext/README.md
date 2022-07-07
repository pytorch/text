# Libtorchtext Examples

- [Tokenizer](./tokenizer)

## Build

The example applications in this directory depend on `libtorch` and `libtorch`. If you have a working `PyTorch`, you
already have `libtorch`. Please refer to
[this tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) for the use of `libtorch` and
TorchScript.

`libtorchtext` is the library of torchtext's C++ components without Python components. It is currently not distributed,
and it will be built alongside with the applications.

The following commands will build `libtorchtext` and applications.

```bash
git submodule update
mkdir build
cd build
cmake -GNinja \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      ..
cmake --build .
```

For the usages of each application, refer to the corresponding application directory.
