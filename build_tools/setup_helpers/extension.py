import os
import platform
import subprocess
from pathlib import Path

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension as TorchBuildExtension
)

__all__ = [
    'get_ext_modules',
    'BuildExtension',
]

_ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
_CSRC_DIR = _ROOT_DIR / 'torchtext' / 'csrc'
_TP_BASE_DIR = _ROOT_DIR / 'third_party'
_TP_INSTALL_DIR = _TP_BASE_DIR / 'build'


def _get_eca(debug):
    eca = []
    if platform.system() == "Windows":
        eca += ['/MT']
    if debug:
        eca += ["-O0", "-g"]
    else:
        if platform.system() == "Windows":
            eca += ['-O2']
        else:
            eca += ["-O3"]
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
    return [str(p) for p in _CSRC_DIR.glob('**/*.cpp')]


def _get_include_dirs():
    return [
        str(_CSRC_DIR),
        str(_TP_INSTALL_DIR / 'include'),
    ]


def _get_library_dirs():
    return [
        str(_TP_INSTALL_DIR / 'lib'),
    ]


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
    return [
        'sentencepiece_train',
        'sentencepiece',
    ]


def _build_sentence_piece(debug):
    build_dir = _TP_BASE_DIR / 'sentencepiece' / 'build'
    build_dir.mkdir(exist_ok=True)
    build_env = os.environ.copy()
    if platform.system() == 'Windows':
        shared = 'OFF'
        extra_args = ['-GNinja']
        build_env.setdefault('CC', 'cl')
        build_env.setdefault('CXX', 'cl')
    else:
        shared = 'OFF'
        extra_args = []
    subprocess.run(
        args=['cmake', f'-DSPM_ENABLE_SHARED={shared}', f'-DCMAKE_INSTALL_PREFIX={_TP_INSTALL_DIR}'] + extra_args + ['..'],
        cwd=str(build_dir),
        check=True,
        env=build_env,
    )
    config = 'Debug' if debug else 'Release'
    print(config)
    subprocess.run(
        args=['cmake', '--build', '.', '--target', 'install', '--config', config],
        cwd=str(build_dir),
        check=True,
        env=build_env,
    )


def _configure_third_party(debug):
    _build_sentence_piece(debug)


_EXT_NAME = 'torchtext._torchtext'


def get_ext_modules(debug=False):
    return [
        CppExtension(
            _EXT_NAME,
            _get_srcs(),
            libraries=_get_libraries(),
            include_dirs=_get_include_dirs(),
            library_dirs=_get_library_dirs(),
            extra_compile_args=_get_eca(debug),
            extra_link_args=_get_ela(debug),
        ),
    ]


class BuildExtension(TorchBuildExtension):
    def build_extension(self, ext):
        if ext.name == _EXT_NAME:
            print(self.debug)
            _configure_third_party(self.debug)
        super().build_extension(ext)
