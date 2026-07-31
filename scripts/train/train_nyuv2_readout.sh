#!/usr/bin/env bash
set -euo pipefail
bash scripts/train/run_readout_matrix.sh \
  nyuv2 depth 80 4 dit_hidden_local
