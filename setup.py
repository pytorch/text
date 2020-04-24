#!/usr/bin/env python
import os
import io
import re

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


def check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in set(["ON", "1", "YES", "TRUE", "Y"])


DEBUG = check_env_flag("DEBUG")
IS_WHEEL = check_env_flag("IS_WHEEL")
IS_CONDA = check_env_flag("IS_CONDA")

print("DEBUG:", DEBUG, "IS_WHEEL:", IS_WHEEL, "IS_CONDA:", IS_CONDA)

eca = []
ela = []
if DEBUG:
    if platform.system() == 'Windows':
        ela += ['/DEBUG:FULL']
    else:
        eca += ['-O0', '-g']
        ela += ['-O0', '-g']


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("torchtext", "__init__.py")
long_description = read("README.rst")

pytorch_package_version = os.getenv("PYTORCH_VERSION")

pytorch_package_dep = "torch"
if pytorch_package_version is not None:
    pytorch_package_dep += "==" + pytorch_package_version

ext_modules = [
    CppExtension(
        "_torchtext",
        ["torchtext/csrc/vocab.cpp"],
        libraries=[],
        include_dirs=[],
        extra_compile_args=eca,
        extra_objects=[],
        extra_link_args=ela,
    ),
]

setup_info = dict(
    # Metadata
    name="torchtext",
    version=version,
    author="PyTorch core devs and James Bradbury",
    author_email="jekbradbury@gmail.com",
    url="https://github.com/pytorch/text",
    description="Text utilities and datasets for PyTorch",
    long_description=long_description,
    license="BSD",
    install_requires=[
        "tqdm",
        "requests",
        "torch",
        "numpy",
        "six",
        "sentencepiece",
        pytorch_package_dep,
    ],
    # Package info
    packages=find_packages(exclude=("test", "test.*")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

setup(**setup_info)
