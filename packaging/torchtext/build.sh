#!/usr/bin/env bash
set -ex

git submodule update --init --recursive
python setup.py install --single-version-externally-managed --record=record.txt
