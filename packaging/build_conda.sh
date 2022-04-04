#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="conda"
export NO_CUDA_PACKAGE=1
setup_env 0.13.0
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_visual_studio_constraint
conda install mkl==2021.2.0
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --python "$PYTHON_VERSION" packaging/torchtext
