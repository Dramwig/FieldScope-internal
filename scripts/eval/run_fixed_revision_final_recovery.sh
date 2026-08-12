#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"
: "${FIELDSCOPE_REPOSITORY:?Set FIELDSCOPE_REPOSITORY}"
: "${FIELDSCOPE_TRACKED_CONFIG:?Set FIELDSCOPE_TRACKED_CONFIG}"
: "${FIELDSCOPE_READOUT_CONFIG:?Set FIELDSCOPE_READOUT_CONFIG}"
: "${FIELDSCOPE_RUNTIME_PROFILE:?Set FIELDSCOPE_RUNTIME_PROFILE}"
: "${FIELDSCOPE_READOUT_RUNTIME_PROFILE:?Set FIELDSCOPE_READOUT_RUNTIME_PROFILE}"

repository="$(cd -- "$FIELDSCOPE_REPOSITORY" && pwd -P)"
python_bin="${FIELDSCOPE_PYTHON:-$FIELDSCOPE_ROOT/.venv/bin/python}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
wrapper_root="${FIELDSCOPE_RECOVERY_WRAPPER_ROOT:-$repository/scripts/eval}"
source_tree_sha256="6ef1305effef91b29d432c9d2c1505b4726645531adad72f60299168d7119c4d"
if [[ "$wrapper_root" != /* ]]; then
  wrapper_root="$repository/$wrapper_root"
fi
output_root="$repository/outputs/full_validation/$cache_tag"
main_evidence="$output_root/evidence_decision.json"
causal_output_root="$output_root/causal"
causal_evidence="$causal_output_root/final_evidence_decision.json"
extension_evidence="$output_root/extension/extension_evidence_decision.json"
final_evidence="$output_root/final_evidence_decision.json"
voc_root="$FIELDSCOPE_DATASETS_ROOT/prepared/pascal_voc_2012"
causal_cache_root="$FIELDSCOPE_DATASETS_ROOT/feature_cache/${cache_tag}_causal"

verify_revision() {
  local actual actual_tree
  actual="$(git -C "$repository" rev-parse HEAD)"
  [[ "$actual" == "$FIELDSCOPE_EXPECTED_REVISION" ]] || {
    echo "revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$actual" >&2
    exit 3
  }
  git -C "$repository" symbolic-ref -q --short HEAD >/dev/null && {
    echo "fixed-revision final recovery requires detached HEAD" >&2
    exit 4
  }
  [[ -z "$(git -C "$repository" status --porcelain)" ]] || {
    echo "fixed-revision final recovery requires a clean worktree" >&2
    exit 5
  }
  actual_tree="$({
    cd "$repository"
    "$python_bin" - <<'PY'
from fieldscope.experiments import code_provenance
print(code_provenance()["code_tree_sha256"])
PY
  })"
  [[ "$actual_tree" == "$source_tree_sha256" ]] || {
    echo "code tree mismatch: expected=$source_tree_sha256 actual=$actual_tree" >&2
    exit 6
  }
}

json_verdict() {
  "$python_bin" - "$1" "$FIELDSCOPE_EXPECTED_REVISION" "$source_tree_sha256" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
tree = sys.argv[3]
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("code_revision") != revision
    or payload.get("code_tree_sha256") != tree
    or payload.get("code_dirty") is not False
):
    raise SystemExit(f"evidence provenance mismatch: {path}")
print(payload.get("verdict", ""))
PY
}

verify_revision
cd "$repository"
[[ -f "$main_evidence" ]] || { echo "missing main evidence: $main_evidence" >&2; exit 6; }

# The causal controls are deliberately kept on the registered old wrapper:
# they do not use the large shared readout cache and their negative result must
# remain part of the final machine-readable decision.
mkdir -p "$causal_output_root" "$causal_cache_root"
declare -A causal_configs=(
  [random_flow]="$repository/configs/ablation/random_auraflow_v03.yaml"
  [spatially_shuffled_probe]="$repository/configs/ablation/auraflow_spatially_shuffled_probe.yaml"
  [neutral_prompt]="$repository/configs/ablation/auraflow_neutral_prompt.yaml"
  [unrelated_prompt]="$repository/configs/ablation/auraflow_unrelated_prompt.yaml"
)
main_verdict="$(json_verdict "$main_evidence")"
if [[ "$main_verdict" != "main_tasks_supported_pending_causal_audits" && \
      "$main_verdict" != "limited_or_negative" ]]; then
  echo "unexpected main evidence verdict=$main_verdict" >&2
  exit 7
fi
for variant in random_flow spatially_shuffled_probe neutral_prompt unrelated_prompt; do
  cache_dir="$causal_cache_root/$variant/voc2012_test"
  report_path="$causal_output_root/$variant.json"
  echo "$(date --iso-8601=seconds) extracting causal VOC cache variant=$variant"
  "$python_bin" -m fieldscope.cli extract-dataset \
    --config "${causal_configs[$variant]}" \
    --root "$voc_root" \
    --dataset voc2012 \
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
  --runtime-profile "$FIELDSCOPE_RUNTIME_PROFILE" \
  --random-flow "$causal_output_root/random_flow.json" \
  --spatially-shuffled-probe "$causal_output_root/spatially_shuffled_probe.json" \
  --neutral-prompt "$causal_output_root/neutral_prompt.json" \
  --unrelated-prompt "$causal_output_root/unrelated_prompt.json" \
  --output "$causal_evidence"

causal_verdict="$(json_verdict "$causal_evidence")"
extension_arguments=()
if [[ "$main_verdict" == "main_tasks_supported_pending_causal_audits" ]]; then
  if [[ -z "${FIELDSCOPE_IMAGENET1K_ROOT:-}" ]]; then
    echo "positive main verdict requires FIELDSCOPE_IMAGENET1K_ROOT for extension" >&2
    exit 8
  fi
  # Run the extension in a child environment with separate extraction/readout
  # identities. The old wrapper overwrites FIELDSCOPE_CONFIG, so this recovery
  # wrapper reproduces its audited steps and explicitly passes both configs.
  extension_script="$wrapper_root/run_fixed_revision_extension_recovery.sh"
  [[ -f "$extension_script" ]] || { echo "missing fixed-revision extension wrapper: $extension_script" >&2; exit 9; }
  env \
    FIELDSCOPE_ROOT="$FIELDSCOPE_ROOT" \
    FIELDSCOPE_DATASETS_ROOT="$FIELDSCOPE_DATASETS_ROOT" \
    FIELDSCOPE_EXPECTED_REVISION="$FIELDSCOPE_EXPECTED_REVISION" \
    FIELDSCOPE_REPOSITORY="$repository" \
    FIELDSCOPE_CONFIG="$FIELDSCOPE_TRACKED_CONFIG" \
    FIELDSCOPE_TRACKED_CONFIG="$FIELDSCOPE_TRACKED_CONFIG" \
    FIELDSCOPE_READOUT_CONFIG="$FIELDSCOPE_READOUT_CONFIG" \
    FIELDSCOPE_RUNTIME_PROFILE="$FIELDSCOPE_RUNTIME_PROFILE" \
    FIELDSCOPE_READOUT_RUNTIME_PROFILE="$FIELDSCOPE_READOUT_RUNTIME_PROFILE" \
    FIELDSCOPE_CACHE_TAG="$cache_tag" \
    FIELDSCOPE_PYTHON="$python_bin" \
    FIELDSCOPE_IMAGENET1K_ROOT="$FIELDSCOPE_IMAGENET1K_ROOT" \
    bash "$extension_script"
  extension_arguments=(--extension-evidence "$extension_evidence")
elif [[ "$causal_verdict" != "limited_or_negative" ]]; then
  echo "negative main evidence has unexpected causal verdict=$causal_verdict" >&2
  exit 10
fi

verify_revision
"$python_bin" -m fieldscope.cli audit-final-evidence \
  --main-evidence "$main_evidence" \
  --causal-evidence "$causal_evidence" \
  "${extension_arguments[@]}" \
  --output "$final_evidence"

echo "$(date --iso-8601=seconds) fixed-revision final evidence written: $final_evidence"
