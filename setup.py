#!/usr/bin/env python
import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('torchtext', '__init__.py')
long_description = read('README.rst')

setup_info = dict(
    # Metadata
    name='torchtext',
    version=VERSION,
    author='PyTorch core devs and James Bradbury',
    author_email='jekbradbury@gmail.com',
    url='https://github.com/pytorch/text',
    description='Text utilities and datasets for PyTorch',
    long_description=long_description,
    license='BSD',

    install_requires=[
        'tqdm', 'requests', 'torch', 'numpy', 'sentencepiece'
    ],
    python_requires='>=3.5',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],

    # Package info
    packages=find_packages(exclude=('test', 'test.*')),

    zip_safe=True,
)

setup(**setup_info)
