#!/usr/bin/env bash
set -euo pipefail
python_bin="${FIELDSCOPE_PYTHON:-python}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
representation="${1:-full}"
"$python_bin" -m fieldscope.cli train-cache \
  --config configs/model/auraflow_v03.yaml \
  --cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/auraflow_v03/cifar10_train" \
  --task classification \
  --representation "$representation" \
  --epochs 20
