#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/check_dataset.py
python train.py
