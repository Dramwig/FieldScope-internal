#!/usr/bin/env bash
set -euo pipefail
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
bash scripts/eval/run_dataset_extract.sh \
  voc2012 \
  "$FIELDSCOPE_DATASETS_ROOT/prepared/pascal_voc_2012"
