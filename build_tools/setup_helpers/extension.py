import os
import platform
import subprocess
from pathlib import Path
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import torch
import distutils.sysconfig

from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension, CppExtension

__all__ = [
    "get_ext_modules",
    # "BuildExtension",
    "CMakeBuild",
]

_ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
_CSRC_DIR = _ROOT_DIR / "torchtext" / "csrc"
_TP_BASE_DIR = _ROOT_DIR / "third_party"
_TP_INSTALL_DIR = _TP_BASE_DIR / "build"


def _get_eca(debug):
    eca = []
    if platform.system() == "Windows":
        eca += ["/MT"]
    if debug:
        eca += ["-O0", "-g"]
    else:
        if platform.system() == "Windows":
            eca += ["-O2"]
        else:
            eca += ["-O3", "-fvisibility=hidden"]
    return eca


def _get_ela(debug):
    ela = []
    if debug:
        if platform.system() == "Windows":
            ela += ["/DEBUG:FULL"]
        else:
            ela += ["-O0", "-g"]
    else:
        if platform.system() != "Windows":
            ela += ["-O3"]
    return ela


def _get_srcs():
    return [str(p) for p in _CSRC_DIR.glob("**/*.cpp")]


def _get_include_dirs():
    return [
        str(_CSRC_DIR),
        str(_TP_INSTALL_DIR / "include"),
    ]


def _get_library_dirs():
    return [str(_TP_INSTALL_DIR / "lib"), str(_TP_INSTALL_DIR / "lib64")]


def _get_libraries():
    # NOTE: The order of the library listed bellow matters.
    #
    # For example, the symbol `sentencepiece::unigram::Model` is
    # defined in sentencepiece but UNDEFINED in sentencepiece_train.
    # GCC only remembers the last encountered symbol.
    # Therefore placing 'sentencepiece_train' after 'sentencepiece' cause runtime error.
    #
    # $ nm third_party/build/lib/libsentencepiece_train.a | grep _ZTIN13sentencepiece7unigram5ModelE
    #                  U _ZTIN13sentencepiece7unigram5ModelE
    # $ nm third_party/build/lib/libsentencepiece.a       | grep _ZTIN13sentencepiece7unigram5ModelE
    # 0000000000000000 V _ZTIN13sentencepiece7unigram5ModelE
    return ["sentencepiece_train", "sentencepiece", "re2", "double-conversion"]


def _get_cxx11_abi():
    try:
        import torch

        value = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        value = 0
    return "-D_GLIBCXX_USE_CXX11_ABI=" + str(value)


# def _build_third_party(debug):
#     build_dir = _TP_BASE_DIR / "build"
#     build_dir.mkdir(exist_ok=True)
#     build_env = os.environ.copy()
#     config = "Debug" if debug else "Release"
#     if platform.system() == "Windows":
#         extra_args = [
#             "-GNinja",
#         ]
#         build_env.setdefault("CC", "cl")
#         build_env.setdefault("CXX", "cl")
#     else:
#         extra_args = ["-DCMAKE_CXX_FLAGS=-fPIC " + _get_cxx11_abi()]
#     subprocess.run(
#         args=[
#             "cmake",
#             "-DBUILD_SHARED_LIBS=OFF",
#             "-DRE2_BUILD_TESTING=OFF",
#             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
#             f"-DCMAKE_INSTALL_PREFIX={_TP_INSTALL_DIR}",
#             f"-DCMAKE_BUILD_TYPE={config}",
#             "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
#             "-DCMAKE_POLICY_DEFAULT_CMP0063=NEW",
#         ]
#         + extra_args
#         + [".."],
#         cwd=str(build_dir),
#         check=True,
#         env=build_env,
#     )
#     print("*** Command list Thirdparty ***")
#     with open(build_dir / "compile_commands.json", "r") as fileobj:
#         print(fileobj.read())
#     print("running cmake --build", flush=True)
#     subprocess.run(
#         args=["cmake", "--build", ".", "--target", "install", "--config", config],
#         cwd=str(build_dir),
#         check=True,
#         env=build_env,
#     )


# def _build_sentence_piece(debug):
#     build_dir = _TP_BASE_DIR / "sentencepiece" / "build"
#     build_dir.mkdir(exist_ok=True)
#     build_env = os.environ.copy()
#     config = "Debug" if debug else "Release"
#     if platform.system() == "Windows":
#         extra_args = ["-GNinja"]
#         build_env.setdefault("CC", "cl")
#         build_env.setdefault("CXX", "cl")
#     else:
#         extra_args = []
#     subprocess.run(
#         args=[
#             "cmake",
#             "-DSPM_ENABLE_SHARED=OFF",
#             f"-DCMAKE_INSTALL_PREFIX={_TP_INSTALL_DIR}",
#             "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
#             "-DCMAKE_CXX_FLAGS=" + _get_cxx11_abi(),
#             "-DCMAKE_POLICY_DEFAULT_CMP0063=NEW",
#             f"-DCMAKE_BUILD_TYPE={config}",
#         ]
#         + extra_args
#         + [".."],
#         cwd=str(build_dir),
#         check=True,
#         env=build_env,
#     )
#     subprocess.run(
#         args=["cmake", "--build", ".", "--target", "install", "--config", config],
#         cwd=str(build_dir),
#         check=True,
#         env=build_env,
#     )


# def _configure_third_party(debug):
#     _build_third_party(debug)
#     _build_sentence_piece(debug)



_LIBTORCHTEXT_NAME = "torchtext.lib.libtorchtext"
_EXT_NAME = "torchtext._torchtext"
_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()

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
            "-DBUILD_TORCHTEXT_PYTHON_EXTENSION:BOOL=ON",
            "-DRE2_BUILD_TESTING:BOOL=OFF",
            "-DBUILD_TESTING:BOOL=OFF"
            # new args
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_POLICY_DEFAULT_CMP0063=NEW",
            "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
            "-DCMAKE_CXX_FLAGS=" + _get_cxx11_abi(),
            "-DSPM_ENABLE_SHARED=OFF",
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
