#!/usr/bin/env bash
set -euo pipefail
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
bash scripts/eval/run_dataset_extract.sh \
  ade20k \
  "$FIELDSCOPE_DATASETS_ROOT/prepared/ade20k"
