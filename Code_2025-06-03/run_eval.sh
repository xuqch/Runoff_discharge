#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/validation.py" \
  --eval_model last \
  --eval_split validation \
  --basins_file 'basins_for_validation.csv' \
  --h5_build_workers 12 \
  --data_dir '/share/home/dq083/Runoff/LSTM/Experiment_for_runoff/Global/' \
  --run_dir '/share/home/dq083/Runoff/LSTM/Experiment_for_runoff/Code_2026-06-03/runs/run_0603_example/' \
  "$@"
