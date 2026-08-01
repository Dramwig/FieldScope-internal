#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_IMAGENET1K_ROOT:?Set FIELDSCOPE_IMAGENET1K_ROOT}"
bash scripts/eval/run_dataset_extract.sh imagenet "$FIELDSCOPE_IMAGENET1K_ROOT"
