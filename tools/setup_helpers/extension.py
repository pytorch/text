import distutils.sysconfig
import os
import platform
import subprocess
from pathlib import Path

import torch
from setuptools import Extension
from setuptools.command.build_ext import build_ext


__all__ = [
    "get_ext_modules",
    "CMakeBuild",
]


_LIBTORCHTEXT_NAME = "torchtext.lib.libtorchtext"
_EXT_NAME = "torchtext._torchtext"
_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()


def _get_cxx11_abi():
    return "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi()))


def get_ext_modules():
    modules = [
        Extension(name=_LIBTORCHTEXT_NAME, sources=[]),
        Extension(name=_EXT_NAME, sources=[]),
    ]
    return modules


# Based off of
# https://github.com/pybind/cmake_example/blob/580c5fd29d4651db99d8874714b07c0c49a53f8a/setup.py


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        # Since two library files (libtorchaudio and _torchaudio) need to be
        # recognized by setuptools, we instantiate `Extension` twice. (see `get_ext_modules`)
        # This leads to the situation where this `build_extension` method is called twice.
        # However, the following `cmake` command will build all of them at the same time,
        # so, we do not need to perform `cmake` twice.
        # Therefore we call `cmake` only for `torchaudio._torchaudio`.
        if ext.name != "torchtext._torchtext":
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            f"-DTORCH_INSTALL_PREFIX:STRING={os.path.dirname(torch.__file__)}",
            "-DBUILD_TORCHTEXT_PYTHON_EXTENSION:BOOL=ON",
            "-DRE2_BUILD_TESTING:BOOL=OFF",
            "-DBUILD_TESTING:BOOL=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_POLICY_DEFAULT_CMP0063=NEW",
            "-DSPM_ENABLE_SHARED=OFF",
            f"-DTORCH_COMPILED_WITH_CXX_ABI={_get_cxx11_abi()}",
        ]
        build_args = ["--target", "install"]

        # Default to Ninja
        if "CMAKE_GENERATOR" not in os.environ or platform.system() == "Windows":
            cmake_args += ["-GNinja"]
        if platform.system() == "Windows":
            import sys

            python_version = sys.version_info
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
            ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split(".")
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = ".".join(without_abi)
        return ext_filename
