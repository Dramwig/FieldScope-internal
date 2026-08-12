#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo \
    "usage: $0 DATASET TASK EPOCHS BATCH_SIZE REFERENCE [NUM_CLASSES] [SEGMENTATION_CLASSES]" \
    >&2
  exit 2
fi

python_bin="${FIELDSCOPE_PYTHON:-python}"
# Keep a dedicated readout override so recovery orchestration never has to
# mutate the config identity embedded in an already-complete feature cache.
config="${FIELDSCOPE_READOUT_CONFIG:-${FIELDSCOPE_CONFIG:-configs/model/auraflow_v03.yaml}}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
dataset="$1"
task="$2"
epochs="$3"
batch_size="$4"
reference="$5"
num_classes="${6:-}"
segmentation_classes="${7:-}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_READOUT_RUNTIME_PROFILE:?Set FIELDSCOPE_READOUT_RUNTIME_PROFILE}"

arguments=(
  --config "$config"
  --train-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/${dataset}_train"
  --val-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/${dataset}_val"
  --test-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/${dataset}_test"
  --output-dir "outputs/full_validation/$cache_tag/$dataset"
  --task "$task"
  --epochs "$epochs"
  --batch-size "$batch_size"
  --learning-rate 0.001
  --weight-decay 0.0001
  --reference "$reference"
  --readout-runtime-profile "$FIELDSCOPE_READOUT_RUNTIME_PROFILE"
)
if [[ -n "$num_classes" ]]; then
  arguments+=(--num-classes "$num_classes")
fi
if [[ -n "$segmentation_classes" ]]; then
  arguments+=(--segmentation-classes "$segmentation_classes")
fi
for representation in \
  random_feature_local z0 zt trajectory velocity mismatch endpoint \
  state state_nograph state_graph \
  response_nograph response_local response \
  full_nograph full_local full \
  dit_hidden_local dit_hidden_attention response_shuffled full_shuffled; do
  arguments+=(--representation "$representation")
done
for seed in 4121 7319 104729; do
  arguments+=(--seed "$seed")
done

"$python_bin" -m fieldscope.cli run-readout-matrix "${arguments[@]}"
