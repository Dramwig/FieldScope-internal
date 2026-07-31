#!/usr/bin/env bash
set -euo pipefail
bash scripts/train/run_readout_matrix.sh \
  ade20k segmentation 80 2 dit_hidden_local 100 150
