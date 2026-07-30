#!/usr/bin/env bash
set -euo pipefail
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
representation="${1:-full}"
python -m fieldscope.cli train-cache \
  --config configs/model/auraflow_v03.yaml \
  --cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/auraflow_v03/voc2012_train" \
  --task segmentation \
  --representation "$representation" \
  --epochs 20

