#!/usr/bin/env python
import distutils.command.clean
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from tools import setup_helpers

ROOT_DIR = Path(__file__).parent.resolve()


def read(*names, **kwargs):
    with io.open(ROOT_DIR.joinpath(*names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def _get_version():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
            version = f.readline().strip()
        if sha is not None:
            version += "+" + sha[:7]

    if sha is None:
        sha = "Unknown"
    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "torchtext" / "version.py"
    with open(version_path, "w") as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))


def _init_submodule():
    print(" --- Initializing submodules")
    try:
        subprocess.check_call(["git", "submodule", "init"])
        subprocess.check_call(["git", "submodule", "update"])
    except Exception:
        print(" --- Submodule initalization failed")
        print("Please run:\n\tgit submodule update --init --recursive")
        sys.exit(1)
    print(" --- Initialized submodule")


VERSION, SHA = _get_version()
_export_version(VERSION, SHA)

print("-- Building version " + VERSION)

pytorch_package_version = os.getenv("PYTORCH_VERSION")
torchdata_package_version = os.getenv("TORCHDATA_VERSION")

pytorch_package_dep = "torch"
if pytorch_package_version is not None:
    pytorch_package_dep += "==" + pytorch_package_version
torchdata_package_dep = "torchdata"
if torchdata_package_version is not None:
    torchdata_package_dep += "==" + torchdata_package_version


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchtext extension
        for path in (ROOT_DIR / "torchtext").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "third_party" / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


_init_submodule()
setup_info = dict(
    # Metadata
    name="torchtext",
    version=VERSION,
    author="PyTorch Text Team",
    author_email="packages@pytorch.org",
    url="https://github.com/pytorch/text",
    description="Text utilities, models, transforms, and datasets for PyTorch.",
    long_description=read("README.rst"),
    license="BSD",
    install_requires=["tqdm", "requests", pytorch_package_dep, "numpy", torchdata_package_dep],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Package info
    packages=find_packages(exclude=("test*", "tools*")),
    zip_safe=False,
    # Extension info
    # If you are trying to use torchtext.so and see no registered op.
    # See here: https://github.com/pytorch/vision/issues/2134"
    ext_modules=setup_helpers.get_ext_modules(),
    cmdclass={
        "build_ext": setup_helpers.CMakeBuild,
        "clean": clean,
    },
)

setup(**setup_info)
