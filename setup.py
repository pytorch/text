#!/usr/bin/env python
import os
import io
import re
import sys
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension

import glob
import shutil
import distutils.command.clean


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


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchtext', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, 'text_extension.cpp'))
    source_core = glob.glob(os.path.join(extensions_dir, 'core', '*.cpp'))

    sources = main_file + source_core
    extension = CppExtension

    define_macros = []
    extra_compile_args = {}

    if sys.platform == 'win32':
        define_macros += [('torchtext_EXPORTS', None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            'torchtext._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


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
        'tqdm', 'requests', 'torch', 'numpy', 'six'
    ],
    
    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension, 'clean': clean},

    # Package info
    packages=find_packages(exclude=('test', 'test.*')),

    zip_safe=True,
)

setup(**setup_info)
