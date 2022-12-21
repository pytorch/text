#!/usr/bin/env bash
set -ex

if [[ ${TARGET_OS} == 'windows' ]]; then
    source /c/Jenkins/Miniconda3/etc/profile.d/conda.sh
else
    eval "$(conda shell.bash hook)"
fi

conda create -y -n ${ENV_NAME} python=${DESIRED_PYTHON} numpy
conda activate ${ENV_NAME}
export CONDA_CHANNEL="pytorch"
export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/cpu"
export PIP_PREFIX=""

if [[ ${CHANNEL} = 'nightly' ]]; then
    export PIP_PREFIX="--pre"
    export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/nightly/cpu"
    export CONDA_CHANNEL="pytorch-nightly"
elif [[ ${CHANNEL} = 'test' ]]; then
    export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/test/cpu"
    export CONDA_CHANNEL="pytorch-test"
fi

if [[ ${PACKAGE_TYPE} = 'conda' ]]; then
    conda install -y torchtext pytorch -c ${CONDA_CHANNEL}
else
    pip3 install --pre torchtext torch --extra-index-url ${PIP_DOWNLOAD_URL}
fi

python3  ./test/smoke_tests/smoke_tests.py
