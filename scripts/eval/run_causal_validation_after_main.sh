#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"

python_bin="${FIELDSCOPE_PYTHON:-python}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
main_evidence="outputs/full_validation/$cache_tag/evidence_decision.json"
causal_output_root="outputs/full_validation/$cache_tag/causal"
causal_cache_root="$FIELDSCOPE_DATASETS_ROOT/feature_cache/${cache_tag}_causal"
voc_root="$FIELDSCOPE_DATASETS_ROOT/prepared/pascal_voc_2012"
runtime_profile="${FIELDSCOPE_RUNTIME_PROFILE:-$PWD/outputs/runtime_gate/auraflow_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json}"

if [[ ! -f "$runtime_profile" ]]; then
  echo "missing runtime profile: $runtime_profile" >&2
  exit 7
fi
export FIELDSCOPE_RUNTIME_PROFILE="$runtime_profile"

actual_revision="$(git rev-parse HEAD)"
if [[ "$actual_revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
  echo "revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$actual_revision" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run causal validation from a dirty worktree" >&2
  exit 4
fi

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
if [[ "$main_verdict" == "incomplete" ]]; then
  echo "main evidence is incomplete; refusing causal validation" >&2
  exit 5
fi
if [[ "$main_verdict" != "main_tasks_supported_pending_causal_audits" && "$main_verdict" != "limited_or_negative" ]]; then
  echo "unexpected main evidence verdict=$main_verdict" >&2
  exit 6
fi
echo "$(date --iso-8601=seconds) running registered causal controls after complete main verdict=$main_verdict"

mkdir -p "$causal_output_root" "$causal_cache_root"
declare -A configs=(
  [random_flow]="configs/ablation/random_auraflow_v03.yaml"
  [spatially_shuffled_probe]="configs/ablation/auraflow_spatially_shuffled_probe.yaml"
  [neutral_prompt]="configs/ablation/auraflow_neutral_prompt.yaml"
  [unrelated_prompt]="configs/ablation/auraflow_unrelated_prompt.yaml"
)

for variant in random_flow spatially_shuffled_probe neutral_prompt unrelated_prompt; do
  cache_dir="$causal_cache_root/$variant/voc2012_test"
  report_path="$causal_output_root/$variant.json"
  echo "$(date --iso-8601=seconds) extracting causal VOC cache variant=$variant"
  "$python_bin" -m fieldscope.cli extract-dataset \
    --config "${configs[$variant]}" \
    --dataset voc2012 \
    --root "$voc_root" \
    --split test \
    --output "$cache_dir" \
    --storage-policy dense \
    --resume
  "$python_bin" -m fieldscope.cli diagnose-segmentation \
    --cache-dir "$cache_dir" \
    --output "$report_path"
done

"$python_bin" -m fieldscope.cli audit-causal-evidence \
  --main-evidence "$main_evidence" \
  --runtime-profile "$runtime_profile" \
  --random-flow "$causal_output_root/random_flow.json" \
  --spatially-shuffled-probe "$causal_output_root/spatially_shuffled_probe.json" \
  --neutral-prompt "$causal_output_root/neutral_prompt.json" \
  --unrelated-prompt "$causal_output_root/unrelated_prompt.json" \
  --output "$causal_output_root/final_evidence_decision.json"

"$python_bin" - "$causal_output_root/final_evidence_decision.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"final causal evidence verdict={payload['verdict']}")
PY
