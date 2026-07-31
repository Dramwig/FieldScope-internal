#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_CHECKPOINTS_ROOT:?Set FIELDSCOPE_CHECKPOINTS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"

python_bin="${FIELDSCOPE_PYTHON:-python}"
signal_root="${FIELDSCOPE_SIGNAL_ROOT:-outputs/signal_gate}"
signal_cache_root="${FIELDSCOPE_SIGNAL_CACHE_ROOT:-$FIELDSCOPE_DATASETS_ROOT/feature_cache/signal_final512}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
preflight_root="outputs/full_validation/$cache_tag/preflight"
decision="$signal_root/promotion_decision.json"
cache_root="$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag"
runtime_profile="${FIELDSCOPE_RUNTIME_PROFILE:-$PWD/outputs/runtime_gate/auraflow_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json}"

if [[ ! -f "$runtime_profile" ]]; then
  echo "missing runtime profile: $runtime_profile" >&2
  exit 11
fi
export FIELDSCOPE_RUNTIME_PROFILE="$runtime_profile"

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
    echo "refusing to run full validation from a dirty worktree" >&2
    exit 4
  fi
}

verify_revision
mkdir -p "$preflight_root" "$cache_root"

"$python_bin" -m fieldscope.cli audit-backbone-assets \
  --config configs/model/auraflow_v03.yaml \
  --output "$preflight_root/auraflow_backbone_asset.json"

if [[ ! -f "$decision" ]]; then
  echo "missing signal-gate decision: $decision" >&2
  exit 5
fi

read -r verdict decision_revision < <(
  "$python_bin" - "$decision" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("verdict", ""), payload.get("code_revision", ""))
PY
)
if [[ "$verdict" == "incomplete" ]]; then
  echo "signal gate is incomplete; refusing full validation" >&2
  exit 6
fi
if [[ "$verdict" != "proceed" && "$verdict" != "stop_or_redesign" ]]; then
  echo "unexpected signal-gate verdict=$verdict" >&2
  exit 8
fi
if [[ "$decision_revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
  echo \
    "signal-gate revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION decision=$decision_revision" \
    >&2
  exit 7
fi
echo \
  "$(date --iso-8601=seconds) signal gate recorded verdict=$verdict; running required full validation"

declare -A dataset_roots=(
  [cifar10]="$FIELDSCOPE_DATASETS_ROOT/prepared/cifar10"
  [voc2012]="$FIELDSCOPE_DATASETS_ROOT/prepared/pascal_voc_2012"
  [imagenet100]="$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet100"
  [ade20k]="$FIELDSCOPE_DATASETS_ROOT/prepared/ade20k"
  [nyuv2]="$FIELDSCOPE_DATASETS_ROOT/prepared/nyuv2"
)
declare -A dataset_counts=(
  [cifar10]="train=45000 val=5000 test=10000"
  [voc2012]="train=1318 val=146 test=1449"
  [imagenet100]="train=116455 val=12940 test=5000"
  [ade20k]="train=18189 val=2021 test=2000"
  [nyuv2]="train=715 val=80 test=654"
)

for dataset in cifar10 voc2012 imagenet100 ade20k nyuv2; do
  arguments=(
    --dataset "$dataset"
    --root "${dataset_roots[$dataset]}"
    --image-size 512
    --output "$preflight_root/${dataset}_split_audit.json"
  )
  if [[ "$dataset" == "imagenet100" ]]; then
    arguments+=(--classes-file "${dataset_roots[$dataset]}/classes.txt")
  fi
  for expected_split in ${dataset_counts[$dataset]}; do
    arguments+=(--expected-split "$expected_split")
  done
  "$python_bin" -m fieldscope.cli audit-dataset-splits "${arguments[@]}"
done

target_split_arguments=()
target_payload_arguments=()
for dataset in voc2012 imagenet100 ade20k nyuv2; do
  while IFS= read -r target; do
    target_split_arguments+=(--target-split "$target")
  done < <(
    "$python_bin" - "$preflight_root/${dataset}_split_audit.json" "$dataset" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit(f"split audit did not pass: {sys.argv[1]}")
for split in payload["split_order"]:
    print(f"{sys.argv[2]}_{split}={payload['splits'][split]['count']}")
PY
  )
done
for split in train val test; do
  target_payload_arguments+=(
    --split-extra-bytes-per-sample "voc2012_${split}=262144"
    --split-extra-bytes-per-sample "ade20k_${split}=262144"
    --split-extra-bytes-per-sample "nyuv2_${split}=1048576"
  )
done

"$python_bin" -m fieldscope.cli plan-cache-budget \
  --measurement-cache "$signal_cache_root/cifar10_train" \
  "${target_split_arguments[@]}" \
  "${target_payload_arguments[@]}" \
  --filesystem-path "$cache_root" \
  --storage-policy readout_sparse \
  --reserve-gib 10 \
  --safety-factor 1.15 \
  --output "$preflight_root/sparse_cache_budget.json"

sparse_projected_bytes="$(
  "$python_bin" - "$preflight_root/sparse_cache_budget.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["projected_total_bytes"])
PY
)"
checkpoint_and_report_budget_bytes=13958643712
voc_test_count="$(
  "$python_bin" - "$preflight_root/voc2012_split_audit.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["splits"]["test"]["count"])
PY
)"

"$python_bin" -m fieldscope.cli plan-cache-budget \
  --measurement-cache "$signal_cache_root/voc2012_test" \
  --target-split "voc2012_dense_main=$voc_test_count" \
  --target-split "voc2012_dense_random_flow=$voc_test_count" \
  --target-split "voc2012_dense_spatially_shuffled_probe=$voc_test_count" \
  --target-split "voc2012_dense_neutral_prompt=$voc_test_count" \
  --target-split "voc2012_dense_unrelated_prompt=$voc_test_count" \
  --filesystem-path "$cache_root" \
  --storage-policy dense \
  --additional-required-bytes "$((sparse_projected_bytes + checkpoint_and_report_budget_bytes))" \
  --reserve-gib 10 \
  --safety-factor 1.15 \
  --output "$preflight_root/combined_cache_budget.json"

fits="$(
  "$python_bin" - "$preflight_root/combined_cache_budget.json" <<'PY'
import json
import sys

print(str(json.load(open(sys.argv[1], encoding="utf-8"))["fits"]).lower())
PY
)"
if [[ "$fits" != "true" ]]; then
  echo \
    "full validation is resource_blocked; see $preflight_root/combined_cache_budget.json" \
    >&2
  exit 8
fi

free_checks=0
while (( free_checks < 5 )); do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    grep -Eq '^[[:space:]]*[0-9]+'; then
    free_checks=0
    echo "$(date --iso-8601=seconds) full validation waiting for free GPU"
  else
    free_checks=$((free_checks + 1))
    echo "$(date --iso-8601=seconds) full validation GPU free check $free_checks/5"
  fi
  sleep 60
done

verify_revision
export FIELDSCOPE_CONFIG="configs/model/auraflow_v03.yaml"
export FIELDSCOPE_CACHE_TAG="$cache_tag"
export FIELDSCOPE_STORAGE_POLICY="readout_sparse"

echo "$(date --iso-8601=seconds) extracting full dense VOC diagnostic cache"
"$python_bin" -m fieldscope.cli extract-dataset \
  --config "$FIELDSCOPE_CONFIG" \
  --dataset voc2012 \
  --root "${dataset_roots[voc2012]}" \
  --split test \
  --output "$FIELDSCOPE_DATASETS_ROOT/feature_cache/${cache_tag}_dense/voc2012_test" \
  --storage-policy dense \
  --resume
"$python_bin" -m fieldscope.cli diagnose-segmentation \
  --cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/${cache_tag}_dense/voc2012_test" \
  --output "outputs/full_validation/$cache_tag/voc2012_unsupervised.json"

for dataset in imagenet100 voc2012 ade20k nyuv2; do
  echo "$(date --iso-8601=seconds) extracting full sparse cache dataset=$dataset"
  bash "scripts/eval/run_${dataset}_extract.sh"
  echo "$(date --iso-8601=seconds) training full readout matrix dataset=$dataset"
  bash "scripts/train/train_${dataset}_readout.sh"
done

"$python_bin" -m fieldscope.cli audit-full-evidence \
  --imagenet100-matrix "outputs/full_validation/$cache_tag/imagenet100/matrix_report.json" \
  --voc2012-matrix "outputs/full_validation/$cache_tag/voc2012/matrix_report.json" \
  --ade20k-matrix "outputs/full_validation/$cache_tag/ade20k/matrix_report.json" \
  --nyuv2-matrix "outputs/full_validation/$cache_tag/nyuv2/matrix_report.json" \
  --voc-unsupervised "outputs/full_validation/$cache_tag/voc2012_unsupervised.json" \
  --backbone-asset "$preflight_root/auraflow_backbone_asset.json" \
  --runtime-profile "$runtime_profile" \
  --imagenet100-split-audit "$preflight_root/imagenet100_split_audit.json" \
  --voc2012-split-audit "$preflight_root/voc2012_split_audit.json" \
  --ade20k-split-audit "$preflight_root/ade20k_split_audit.json" \
  --nyuv2-split-audit "$preflight_root/nyuv2_split_audit.json" \
  --output "outputs/full_validation/$cache_tag/evidence_decision.json"

verdict="$(
  "$python_bin" - "outputs/full_validation/$cache_tag/evidence_decision.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["verdict"])
PY
)"
echo "$(date --iso-8601=seconds) full validation evidence verdict=$verdict"
if [[ "$verdict" == "incomplete" ]]; then
  echo "main evidence is incomplete; refusing final causal audit" >&2
  exit 9
fi
if [[ "$verdict" != "main_tasks_supported_pending_causal_audits" && "$verdict" != "limited_or_negative" ]]; then
  echo "unexpected main evidence verdict=$verdict" >&2
  exit 10
fi
exec bash scripts/eval/run_causal_validation_after_main.sh
