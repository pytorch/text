#!/usr/bin/env python
import io
import os
import shutil
import subprocess
from pathlib import Path
import distutils.command.clean
from setuptools import setup, find_packages

from build_tools import setup_helpers

ROOT_DIR = Path(__file__).parent.resolve()


def read(*names, **kwargs):
    with io.open(ROOT_DIR.joinpath(*names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def _get_version():
    version = '0.8.0a0'
    sha = None

    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        sha = subprocess.check_output(cmd, cwd=str(ROOT_DIR)).decode('ascii').strip()
    except Exception:
        pass

    if os.getenv('BUILD_VERSION'):
        version = os.getenv('BUILD_VERSION')
    elif sha is not None:
        version += '+' + sha[:7]

    if sha is None:
        sha = 'Unknown'
    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / 'torchtext' / 'version.py'
    with open(version_path, 'w') as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))


VERSION, SHA = _get_version()
_export_version(VERSION, SHA)

print('-- Building version ' + VERSION)


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchtext extension
        for path in (ROOT_DIR / 'torchtext').glob('**/*.so'):
            print(f'removing \'{path}\'')
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / 'build',
            ROOT_DIR / 'third_party' / 'build',
        ]
        for path in build_dirs:
            if path.exists():
                print(f'removing \'{path}\' (and everything under it)')
                shutil.rmtree(str(path), ignore_errors=True)


setup_info = dict(
    # Metadata
    name='torchtext',
    version=VERSION,
    author='PyTorch core devs and James Bradbury',
    author_email='jekbradbury@gmail.com',
    url='https://github.com/pytorch/text',
    description='Text utilities and datasets for PyTorch',
    long_description=read('README.rst'),
    license='BSD',

    install_requires=[
        'tqdm', 'requests', 'torch', 'numpy'
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
    packages=find_packages(exclude=('test*', 'build_tools*')),
    zip_safe=False,
    # Extension info
    # If you are trying to use torchtext.so and see no registered op.
    # See here: https://github.com/pytorch/vision/issues/2134"
    ext_modules=setup_helpers.get_ext_modules(),
    cmdclass={
        'build_ext': setup_helpers.BuildExtension.with_options(no_python_abi_suffix=True),
        'clean': clean,
    },
)

setup(**setup_info)
