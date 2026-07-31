#!/usr/bin/env bash
set -euo pipefail
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
root="$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet100"
bash scripts/eval/run_dataset_extract.sh imagenet100 "$root" "$root/classes.txt"
