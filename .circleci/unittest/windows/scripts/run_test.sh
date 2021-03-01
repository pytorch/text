#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

python -m torch.utils.collect_env
cd test
pytest --cov=torchtext --junitxml=test-results/junit.xml -v --durations 20 .
