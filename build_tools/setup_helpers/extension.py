import os
import platform
import subprocess
from pathlib import Path
import distutils.sysconfig

import torch
from setuptools import Extension
from setuptools.command.build_ext import build_ext

__all__ = [
    'get_ext_modules',
    'CMakeBuild',
]

_ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
_TP_BASE_DIR = _ROOT_DIR / 'third_party'


def get_ext_modules():
    return [Extension(name='torchtext._torchtext', sources=[])]


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake is not available.")
        super().run()

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            '-DCMAKE_VERBOSE_MAKEFILE=ON',
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            "-DSPM_ENABLE_SHARED=OFF",
            "-DSPM_BUILD_TEST=OFF",
            "-DSPM_COVERAGE=OFF",
            "-DSPM_ENABLE_NFKC_COMPILE=OFF",
            "-DRE2_BUILD_TESTING=OFF",
            "-DBUILD_TESTING=OFF",
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        ]

        if platform.system() == 'Windows':
            cmake_args.extend([
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
            ])
        build_args = [
            '--target', 'install', '--config', cfg
        ]

        # Default to Ninja
        if 'CMAKE_GENERATOR' not in os.environ:
            cmake_args += ["-GNinja"]

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

        subprocess.check_call(
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split('.')
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = '.'.join(without_abi)
        return ext_filename
