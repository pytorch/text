
if [[ ${MATRIX_PACKAGE_TYPE} = "conda" ]]; then
    conda install -y torchtext==0.17.2 -c ${PYTORCH_CONDA_CHANNEL}
else
    pip install ${PYTORCH_PIP_PREFIX} torchtext==0.17.2 --index-url ${PYTORCH_PIP_DOWNLOAD_URL}
fi

python  ./test/smoke_tests/smoke_tests.py
