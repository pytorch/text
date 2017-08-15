#!/usr/bin/env python
from setuptools import setup, find_packages

VERSION = '0.2.0a'

long_description = '''torch-text provides text and NLP data utilities
and datasets for torch'''

setup_info = dict(
    # Metadata
    name='torchtext',
    version=VERSION,
    author='PyTorch core devs and James Bradbury',
    author_email='jekbradbury@gmail.com',
    url='https://github.com/pytorch/text',
    description='text utilities and datasets for torch deep learning',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
)

setup(**setup_info)
