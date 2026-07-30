#!/usr/bin/env bash
set -euo pipefail
python_bin="${FIELDSCOPE_PYTHON:-python}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
"$python_bin" -m fieldscope.cli extract-dataset \
  --config configs/model/auraflow_v03.yaml \
  --dataset cifar10 \
  --root "$FIELDSCOPE_DATASETS_ROOT/prepared/cifar10" \
  --split train \
  --output "$FIELDSCOPE_DATASETS_ROOT/feature_cache/auraflow_v03/cifar10_train"
