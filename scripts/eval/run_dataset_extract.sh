#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATASET PREPARED_ROOT [CLASSES_FILE]" >&2
  exit 2
fi

python_bin="${FIELDSCOPE_PYTHON:-python}"
# Extraction and cached-readout execution have different provenance contracts.
# A fixed-revision recovery may therefore keep extraction on the tracked config
# while training uses a content-addressed, lower-memory runtime-only config.
config="${FIELDSCOPE_EXTRACTION_CONFIG:-${FIELDSCOPE_CONFIG:-configs/model/auraflow_v03.yaml}}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
storage_policy="${FIELDSCOPE_STORAGE_POLICY:-readout_sparse}"
dataset="$1"
prepared_root="$2"
classes_file="${3:-}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"

class_arguments=()
if [[ -n "$classes_file" ]]; then
  class_arguments=(--classes-file "$classes_file")
fi

for split in train val test; do
  "$python_bin" -m fieldscope.cli extract-dataset \
    --config "$config" \
    --dataset "$dataset" \
    --root "$prepared_root" \
    --split "$split" \
    --output "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/${dataset}_${split}" \
    --storage-policy "$storage_policy" \
    --resume \
    "${class_arguments[@]}"
done
