#!/usr/bin/env bash
export PYTHONPATH="$PWD"
set -e
source .venv/bin/activate
python scripts/check_dataset.py
python train.py
