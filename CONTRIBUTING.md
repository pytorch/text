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
