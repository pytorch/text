conda create -y -n ${ENV_NAME} python=${MATRIX_PYTHON_VERSION} numpy
conda activate ${ENV_NAME}
export CONDA_CHANNEL="pytorch"
export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/cpu"
export TEXT_PIP_PREFIX=""

if [[ ${MATRIX_CHANNEL} = 'nightly' ]]; then
    export TEXT_PIP_PREFIX="--pre"
    export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/nightly/cpu"
    export CONDA_CHANNEL="pytorch-nightly"
elif [[ ${MATRIX_CHANNEL} = 'test' ]]; then
    export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/test/cpu"
    export CONDA_CHANNEL="pytorch-test"
fi

conda install -y  pytorch -c ${CONDA_CHANNEL}
pip install ${TEXT_PIP_PREFIX} torchtext torch --extra-index-url ${PIP_DOWNLOAD_URL}

python  ./test/smoke_tests/smoke_tests.py
