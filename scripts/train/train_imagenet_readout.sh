#!/usr/bin/env bash
set -euo pipefail

bash scripts/train/run_readout_matrix.sh \
  imagenet classification 90 128 dit_hidden_local 1000
