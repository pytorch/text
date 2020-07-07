#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
export NO_CUDA_PACKAGE=1
setup_env 0.8.0
setup_wheel_python
pip_install numpy future
setup_pip_pytorch_version
git submodule update --init --recursive
python setup.py clean
if [[ "$OSTYPE" == "msys" ]]; then
    "$script_dir/vc_env_helper.bat" python setup.py bdist_wheel
else
    python setup.py bdist_wheel
fi
