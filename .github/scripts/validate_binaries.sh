
if [[ ${MATRIX_PACKAGE_TYPE} = "conda" ]]; then
    #special case for Python 3.11
    if [[ ${MATRIX_PYTHON_VERSION} == '3.11' ]]; then
        conda install -y torchtext -c malfet -c ${PYTORCH_CONDA_CHANNEL}
    else
        conda install -y torchtext -c ${PYTORCH_CONDA_CHANNEL}
    fi
else
    export filedownload=$(pip install ${PYTORCH_PIP_PREFIX} torchtext --index-url ${PYTORCH_PIP_DOWNLOAD_URL} | grep Downloading.*torchtext.* | grep -Eio '\bhttps://.*whl\b')
    curl -O filedownload
    unzip torchtext*
    cd $(ls | grep -P "torchtext.*dist-info")
    export match=$(cat METADATA | grep "torch (==2.0.0)")
    [ -z "$var" ] && exit(1)
    #pip install ${PYTORCH_PIP_PREFIX} torchtext --index-url ${PYTORCH_PIP_DOWNLOAD_URL}
fi

python  ./test/smoke_tests/smoke_tests.py
