#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version

run_tests() {
    py.test -v --cov=./
}

if [[ "$RUN_FLAKE8" == "true" ]]; then
    flake8
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi
