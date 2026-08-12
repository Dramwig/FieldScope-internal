#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"
: "${FIELDSCOPE_IMAGENET1K_ROOT:?Set FIELDSCOPE_IMAGENET1K_ROOT}"

python_bin="${FIELDSCOPE_PYTHON:-python}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
tracked_config="${FIELDSCOPE_EXTRACTION_CONFIG:-${FIELDSCOPE_CONFIG:-configs/model/auraflow_v03.yaml}}"
readout_config="${FIELDSCOPE_READOUT_CONFIG:-${FIELDSCOPE_CONFIG:-configs/model/auraflow_v03.yaml}}"
cache_root="$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag"
extension_cache_root="$FIELDSCOPE_DATASETS_ROOT/feature_cache/${cache_tag}_extension"
output_root="outputs/full_validation/$cache_tag/extension"
preflight_root="$output_root/preflight"
main_evidence="outputs/full_validation/$cache_tag/evidence_decision.json"
main_runtime_profile="${FIELDSCOPE_RUNTIME_PROFILE:-$PWD/outputs/runtime_gate/auraflow_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json}"
readout_runtime_profile="${FIELDSCOPE_READOUT_RUNTIME_PROFILE:-$PWD/outputs/runtime_gate/readout_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json}"
imagenet_manifest="$(dirname "$FIELDSCOPE_IMAGENET1K_ROOT")/metadata/image_manifest.jsonl"
imagenet_summary="$(dirname "$FIELDSCOPE_IMAGENET1K_ROOT")/metadata/export_summary.json"
resource_poll_seconds="${FIELDSCOPE_EXTENSION_RESOURCE_POLL_SECONDS:-600}"

if ! [[ "$resource_poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIELDSCOPE_EXTENSION_RESOURCE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

verify_revision() {
  local actual_revision
  actual_revision="$(git rev-parse HEAD)"
  if [[ "$actual_revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
    echo "revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$actual_revision" >&2
    exit 3
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing extension validation from a dirty worktree" >&2
    exit 4
  fi
}

verify_revision
mkdir -p "$preflight_root" "$extension_cache_root"
main_verdict="$(
  "$python_bin" - "$main_evidence" "$FIELDSCOPE_EXPECTED_REVISION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("code_revision") != sys.argv[2] or payload.get("code_dirty") is not False:
    raise SystemExit("main evidence provenance mismatch")
print(payload.get("verdict", ""))
PY
)"
if [[ "$main_verdict" != "main_tasks_supported_pending_causal_audits" ]]; then
  echo "conditional extension requires positive main evidence; got $main_verdict" >&2
  exit 5
fi

"$python_bin" -m fieldscope.cli audit-imagenet1k-asset \
  --root "$FIELDSCOPE_IMAGENET1K_ROOT" \
  --manifest "$imagenet_manifest" \
  --export-summary "$imagenet_summary" \
  --output "$preflight_root/imagenet1k_asset_audit.json"

mapfile -t imagenet_counts < <(
  "$python_bin" - "$preflight_root/imagenet1k_asset_audit.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit("ImageNet-1k asset audit did not pass")
for split in ("train", "val", "test"):
    print(f"{split}={payload['fieldscope_split_counts'][split]}")
PY
)
split_arguments=()
for specification in "${imagenet_counts[@]}"; do
  split_arguments+=(--expected-split "$specification")
done
"$python_bin" -m fieldscope.cli audit-dataset-splits \
  --dataset imagenet \
  --root "$FIELDSCOPE_IMAGENET1K_ROOT" \
  --image-size 512 \
  "${split_arguments[@]}" \
  --output "$preflight_root/imagenet1k_split_audit.json"

"$python_bin" -m fieldscope.cli build-extension-ablation-configs \
  --base-config "$tracked_config" \
  --output-dir "$preflight_root/ablation_configs" \
  --output "$preflight_root/ablation_config_registry.json"

target_split_arguments=()
for specification in "${imagenet_counts[@]}"; do
  target_split_arguments+=(--target-split "imagenet_${specification}")
done
imagenet_checkpoint_budget_bytes=8589934592
"$python_bin" -m fieldscope.cli plan-cache-budget \
  --measurement-cache "$cache_root/imagenet100_train" \
  "${target_split_arguments[@]}" \
  --filesystem-path "$extension_cache_root" \
  --storage-policy readout_sparse \
  --additional-required-bytes "$imagenet_checkpoint_budget_bytes" \
  --reserve-gib 10 \
  --safety-factor 1.15 \
  --output "$preflight_root/imagenet1k_sparse_cache_budget.json"
imagenet_projected_bytes="$(
  "$python_bin" - "$preflight_root/imagenet1k_sparse_cache_budget.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["projected_total_bytes"])
PY
)"
voc_test_count=1449
ablation_target_arguments=()
while IFS= read -r name; do
  ablation_target_arguments+=(--target-split "voc_${name}=$voc_test_count")
done < <(
  "$python_bin" - "$preflight_root/ablation_config_registry.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for variant in payload["variants"]:
    print(variant["name"])
PY
)
write_combined_extension_budget() {
  "$python_bin" -m fieldscope.cli plan-cache-budget \
    --measurement-cache "$FIELDSCOPE_DATASETS_ROOT/feature_cache/${cache_tag}_dense/voc2012_test" \
    "${ablation_target_arguments[@]}" \
    --filesystem-path "$extension_cache_root" \
    --storage-policy dense \
    --additional-required-bytes \
      "$((imagenet_projected_bytes + imagenet_checkpoint_budget_bytes))" \
    --reserve-gib 10 \
    --safety-factor 1.15 \
    --output "$preflight_root/combined_extension_cache_budget.json"
}

while true; do
  write_combined_extension_budget
  fits="$(
    "$python_bin" - "$preflight_root/combined_extension_cache_budget.json" <<'PY'
import json
import sys
print(str(json.load(open(sys.argv[1], encoding="utf-8"))["fits"]).lower())
PY
  )"
  if [[ "$fits" == "true" ]]; then
    break
  fi
  echo \
    "$(date --iso-8601=seconds) extension is resource_blocked; refusing to shrink the registered scope; waiting ${resource_poll_seconds}s" \
    >&2
  sleep "$resource_poll_seconds"
  verify_revision
done

verify_revision
export FIELDSCOPE_CONFIG="$tracked_config"
export FIELDSCOPE_EXTRACTION_CONFIG="$tracked_config"
export FIELDSCOPE_READOUT_CONFIG="$readout_config"
export FIELDSCOPE_RUNTIME_PROFILE="$main_runtime_profile"
export FIELDSCOPE_READOUT_RUNTIME_PROFILE="$readout_runtime_profile"
export FIELDSCOPE_STORAGE_POLICY="readout_sparse"
export FIELDSCOPE_CACHE_TAG="$cache_tag"
bash scripts/eval/run_imagenet_extract.sh
bash scripts/train/train_imagenet_readout.sh

ablation_report_arguments=()
ablation_profile_arguments=()
while IFS=$'\t' read -r name config_path; do
  profile_path="$preflight_root/ablation_runtime_profiles/${name}.json"
  report_path="$output_root/ablations/${name}.json"
  cache_dir="$extension_cache_root/ablations/${name}/voc2012_test"
  mkdir -p "$(dirname "$profile_path")" "$(dirname "$report_path")"
  unset FIELDSCOPE_RUNTIME_PROFILE
  "$python_bin" -m fieldscope.cli runtime-gate \
    --config "$config_path" \
    --output "$profile_path"
  export FIELDSCOPE_RUNTIME_PROFILE="$profile_path"
  "$python_bin" -m fieldscope.cli extract-dataset \
    --config "$config_path" \
    --dataset voc2012 \
    --root "$FIELDSCOPE_DATASETS_ROOT/prepared/pascal_voc_2012" \
    --split test \
    --output "$cache_dir" \
    --storage-policy dense \
    --resume
  "$python_bin" -m fieldscope.cli diagnose-segmentation \
    --cache-dir "$cache_dir" \
    --output "$report_path"
  ablation_report_arguments+=(--ablation-report "$name=$report_path")
  ablation_profile_arguments+=(--ablation-runtime-profile "$name=$profile_path")
done < <(
  "$python_bin" - "$preflight_root/ablation_config_registry.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for variant in payload["variants"]:
    print(f"{variant['name']}\t{variant['config_path']}")
PY
)

export FIELDSCOPE_RUNTIME_PROFILE="$main_runtime_profile"
verify_revision
"$python_bin" -m fieldscope.cli audit-extension-evidence \
  --main-evidence "$main_evidence" \
  --imagenet-matrix "outputs/full_validation/$cache_tag/imagenet/matrix_report.json" \
  --imagenet-asset-audit "$preflight_root/imagenet1k_asset_audit.json" \
  --imagenet-split-audit "$preflight_root/imagenet1k_split_audit.json" \
  --main-runtime-profile "$main_runtime_profile" \
  --readout-runtime-profile "$readout_runtime_profile" \
  --base-voc-report "outputs/full_validation/$cache_tag/voc2012_unsupervised.json" \
  --base-config "$tracked_config" \
  --ablation-registry "$preflight_root/ablation_config_registry.json" \
  "${ablation_report_arguments[@]}" \
  "${ablation_profile_arguments[@]}" \
  --output "$output_root/extension_evidence_decision.json"
