#!/usr/bin/env bash
set -euo pipefail
bash scripts/train/run_readout_matrix.sh \
  cifar10 classification 50 128 dit_hidden_local 10
