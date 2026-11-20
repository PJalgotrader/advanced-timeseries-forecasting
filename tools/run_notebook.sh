#!/usr/bin/env bash
# Runs a notebook from top to bottom and saves it with its outputs, in place.
#   bash tools/run_notebook.sh 07-deep-sequence-models/01_rnn_lstm_one_series.ipynb [python] [kernel]
# python: the Python of the environment to run it in (default: the one on the PATH). It needs keras with one backend,
# numpy, pandas, matplotlib, and jupyter's nbconvert and ipykernel.
# kernel: the Jupyter kernel name of that environment (default python3). Check the first cell's output afterwards:
# a machine with several Pythons can hand "python3" to another one, with another Keras.
set -euo pipefail
NB="$1"; PY="${2:-python}"; KERNEL="${3:-python3}"
cd "$(dirname "$NB")"
"$PY" -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name="$KERNEL" --ExecutePreprocessor.timeout=-1 "$(basename "$NB")"
