#!/usr/bin/env bash

set -u

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# We want to run all the style checks even if one of them fail.

exit_status=0

printf "\x1b[34mRunning flake8: "
flake8 --version
printf "\x1b[0m\n"
flake8 torchtext test build_tools/setup_helpers
status=$?
exit_status="$((exit_status+status))"
if [ "${status}" -ne 0 ]; then
    printf "\x1b[31mflake8 failed. Check the format of Python files.\x1b[0m\n"
fi

printf "\x1b[34mRunning clang-format: "
clang-format --version
printf "\x1b[0m\n"
git-clang-format origin/master
git diff --exit-code
status=$?
exit_status="$((exit_status+status))"
if [ "${status}" -ne 0 ]; then
    printf "\x1b[31mC++ files are not formatted. Please use git-clang-format to format CPP files.\x1b[0m\n"
fi
exit $exit_status
