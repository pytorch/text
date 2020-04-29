#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
flake8
pytest --cov=torchtext --junitxml=test-results/junit.xml -v test
