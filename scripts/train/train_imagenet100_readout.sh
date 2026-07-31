#!/usr/bin/env bash
set -euo pipefail
bash scripts/train/run_readout_matrix.sh \
  imagenet100 classification 90 128 dit_hidden_local 100
