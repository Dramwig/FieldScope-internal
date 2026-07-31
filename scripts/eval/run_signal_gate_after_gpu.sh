#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_CHECKPOINTS_ROOT:?Set FIELDSCOPE_CHECKPOINTS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"

python_bin="${FIELDSCOPE_PYTHON:-python}"
wait_for_pid="${FIELDSCOPE_WAIT_FOR_PID:-}"
config="configs/eval/auraflow_signal_gate.yaml"
cache_root="$FIELDSCOPE_DATASETS_ROOT/feature_cache/signal_final512"
output_root="outputs/signal_gate"
runtime_profile="outputs/runtime_gate/auraflow_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json"
readout_runtime_profile="outputs/runtime_gate/readout_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json"

verify_revision() {
  local actual_revision
  actual_revision="$(git rev-parse HEAD)"
  if [[ "$actual_revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
    echo \
      "revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$actual_revision" \
      >&2
    exit 3
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing to run signal gate from a dirty worktree" >&2
    exit 4
  fi
}

verify_revision

while [[ -n "$wait_for_pid" ]] && kill -0 "$wait_for_pid" 2>/dev/null; do
  echo "$(date --iso-8601=seconds) waiting for external runbook pid=$wait_for_pid"
  sleep 60
done

free_checks=0
while (( free_checks < 5 )); do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    grep -Eq '^[[:space:]]*[0-9]+'; then
    free_checks=0
    echo "$(date --iso-8601=seconds) GPU still occupied"
  else
    free_checks=$((free_checks + 1))
    echo "$(date --iso-8601=seconds) GPU free check $free_checks/5"
  fi
  sleep 60
done

verify_revision
mkdir -p "$cache_root" "$output_root"
mkdir -p "$(dirname "$runtime_profile")"
unset FIELDSCOPE_RUNTIME_PROFILE
"$python_bin" -m fieldscope.cli runtime-gate \
  --config "$config" \
  --output "$runtime_profile"
export FIELDSCOPE_RUNTIME_PROFILE="$PWD/$runtime_profile"
for specification in "train 256" "val 128" "test 128"; do
  read -r split limit <<<"$specification"
  "$python_bin" -m fieldscope.cli extract-dataset \
    --config "$config" \
    --dataset cifar10 \
    --root "$FIELDSCOPE_DATASETS_ROOT/prepared/cifar10" \
    --split "$split" \
    --limit "$limit" \
    --output "$cache_root/cifar10_$split" \
    --storage-policy readout_sparse \
    --resume
done

"$python_bin" -m fieldscope.cli extract-dataset \
  --config "$config" \
  --dataset voc2012 \
  --root "$FIELDSCOPE_DATASETS_ROOT/prepared/pascal_voc_2012" \
  --split test \
  --limit 128 \
  --output "$cache_root/voc2012_test" \
  --resume

"$python_bin" -m fieldscope.cli diagnose-segmentation \
  --cache-dir "$cache_root/voc2012_test" \
  --output "$output_root/voc2012_unsupervised.json"

matrix_arguments=(
  --config "$config"
  --train-cache-dir "$cache_root/cifar10_train"
  --val-cache-dir "$cache_root/cifar10_val"
  --test-cache-dir "$cache_root/cifar10_test"
  --output-dir "$output_root/cifar10_readout"
  --task classification
  --epochs 20
  --batch-size 32
  --learning-rate 0.001
  --weight-decay 0.0001
  --reference dit_hidden_local
  --num-classes 10
)
for representation in \
  z0 trajectory state response_local response full \
  dit_hidden_local dit_hidden_attention response_shuffled full_shuffled; do
  matrix_arguments+=(--representation "$representation")
done
for seed in 4121 7319 104729; do
  matrix_arguments+=(--seed "$seed")
done

"$python_bin" -m fieldscope.cli run-readout-matrix "${matrix_arguments[@]}"

mkdir -p "$(dirname "$readout_runtime_profile")"
"$python_bin" -m fieldscope.cli readout-runtime-gate \
  --config configs/model/auraflow_v03.yaml \
  --train-cache-dir "$cache_root/cifar10_train" \
  --val-cache-dir "$cache_root/cifar10_val" \
  --test-cache-dir "$cache_root/cifar10_test" \
  --output "$readout_runtime_profile"

"$python_bin" -m fieldscope.cli audit-signal-gate \
  --cache-dir "$cache_root/cifar10_train" \
  --cache-dir "$cache_root/cifar10_val" \
  --cache-dir "$cache_root/cifar10_test" \
  --cache-dir "$cache_root/voc2012_test" \
  --voc-report "$output_root/voc2012_unsupervised.json" \
  --cifar-matrix "$output_root/cifar10_readout/matrix_report.json" \
  --output "$output_root/promotion_decision.json"

verdict="$(
  "$python_bin" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["verdict"])' \
    "$output_root/promotion_decision.json"
)"
echo "$(date --iso-8601=seconds) signal gate completed verdict=$verdict"
