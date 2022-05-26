# Contributing to text

We want to make contributing to this project as easy and transparent as possible.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

### Code style

`torchtext` enforces a fairly strict code format for Python, text, and configuration files through
[`pre-commit`](https://pre-commit.com). You can install it with

```shell
pip install pre-commit
```

or

```shell
conda install -c conda-forge pre-commit
```

To check and in most cases fix the code format, stage all your changes (`git add`) and execute `pre-commit run`. To
perform the checks automatically before every `git commit`, you can install the checks as hooks with
`pre-commit install`.

In addition, `torchtext` also enforces a fairly strict code format for C++ files through a custom version of
[`clang-format`](https://clang.llvm.org/docs/ClangFormat.html). You can download it from

- https://oss-clang-format.s3.us-east-2.amazonaws.com/mac/clang-format-mojave
- https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64

depending on your platform. To run the formatter, make the binary executable (`chmod +x`) and execute

```shell
python run-clang-format.py \
    --recursive \
    --clang-format-executable=$CLANG_FORMAT \
    torchtext/csrc
```

where `$CLANG_FORMAT` denotes the path to the downloaded binary.

## Adding Third Party Libraries

The following steps outline how to add third party libraries to torchtext:

1. Add the third party library as a submodule. Here is a great
   [tutorial](https://www.atlassian.com/git/tutorials/git-submodule) on working with submodules in git.
   - Navigate to `third_party/` folder and run `git submodule add <repo-URL>`
   - Verify the newly added module is present in the
     [`.gitmodules`](https://github.com/pytorch/text/blob/main/.gitmodules) file
2. Update
   [`third_party/CMakeLists.txt`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/third_party/CMakeLists.txt#L8)
   to add the following line: `add_subdirectory(<name-of-submodule-folder> EXCLUDE_FROM_ALL)`
3. (Optional) If any of the files within the `csrc/` folder make use of the newly added third party library then
   - Add the new submodule folder to
     [`​​LIBTORCHTEXT_INCLUDE_DIRS`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/csrc/CMakeLists.txt#L24)
     and to
     [`EXTENSION_INCLUDE_DIRS`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/csrc/CMakeLists.txt#L119)
   - Add the "targets" name defined by the third party library's `CMakeLists` file to
     [`LIBTORCHTEXT_LINK_LIBRARIES`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/csrc/CMakeLists.txt#L33)
   - Note that the third party libraries are linked statically with torchtext
4. Verify the torchtext build works by running `python setup.py develop`

## Adding a Custom C++ Operator

Custom C++ operators can be implemented and registered in torchtext for several reasons including to make an existing
Python component more efficient, and to get around the limitations when working with multithreading in Python (due to
the Global Interpreter Lock). These custom kernels (or “ops”) can be embedded into a TorchScripted model and can be
executed both in Python and in their serialized form directly in C++. You can learn more in this
[tutorial on writing custom C++ operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)

Steps to register an operator:

1. Add the new custom operator to the [`torchtext/csrc`](https://github.com/pytorch/text/tree/main/torchtext/csrc)
   folder. This entails writing the header and the source file for the custom op.
2. Add the new source files to the
   [`LIBTORCHTEXT_SOURCES`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/csrc/CMakeLists.txt#L11)
   list.
3. Register the operators with torchbind and pybind
   - Torchbind registration happens in the
     [`register_torchbindings.cpp`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/csrc/register_torchbindings.cpp#L14)
     file
   - Pybind registration happens in the
     [`register_pybindings.cpp`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/csrc/register_pybindings.cpp#L34)
     file.
4. Write a Python wrapper class that is responsible for exposing the torchbind/pybind registered operators via Python.
   You can find some examples of this in the
   [`torchtext/transforms.py`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/torchtext/transforms.py#L274)
   file.
5. Write a unit test that tests the functionality of the operator through the Python wrapper class. You can find some
   examples in the
   [`test/test_transforms.py`](https://github.com/pytorch/text/blob/70fc1040ee40faf129604557107cc59fd51c4fe2/test/test_transforms.py#L317)
   file.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of
Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be
able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe disclosure of security bugs. In those
cases, please go through the process outlined on that page and do not file a public issue.

## License

By contributing to text, you agree that your contributions will be licensed under the LICENSE file in the root directory
of this source tree.
