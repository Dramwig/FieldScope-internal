#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_EXPECTED_SOURCE_REVISION:?Set FIELDSCOPE_EXPECTED_SOURCE_REVISION}"
: "${FIELDSCOPE_EXPECTED_ANALYZER_REVISION:?Set FIELDSCOPE_EXPECTED_ANALYZER_REVISION}"

source_repository="${FIELDSCOPE_SOURCE_REPOSITORY:-$FIELDSCOPE_ROOT/FieldScope-internal}"
analyzer_repository="${FIELDSCOPE_ANALYZER_REPOSITORY:-$PWD}"
python_bin="${FIELDSCOPE_PYTHON:-$FIELDSCOPE_ROOT/.venv/bin/python}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
poll_seconds="${FIELDSCOPE_ERROR_ANALYSIS_POLL_SECONDS:-600}"
output_root="${FIELDSCOPE_ERROR_ANALYSIS_OUTPUT_ROOT:-$FIELDSCOPE_ROOT/../FieldScope-analysis-output/formal-supervised-errors}"
final_decision="$source_repository/outputs/full_validation/$cache_tag/final_evidence_decision.json"
lock_file="$output_root/waiter.lock"
pid_file="$output_root/waiter.pid"

if ! [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIELDSCOPE_ERROR_ANALYSIS_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$output_root"
exec 9>"$lock_file"
if ! flock -n 9; then
  exit 0
fi
printf '%s\n' "$$" >"${pid_file}.tmp.$$"
mv -f "${pid_file}.tmp.$$" "$pid_file"
trap 'rm -f "$pid_file"' EXIT

export PYTHONPATH="$analyzer_repository/src${PYTHONPATH:+:$PYTHONPATH}"

verify_checkout() {
  local repository="$1"
  local expected_revision="$2"
  local name="$3"
  local actual_revision
  actual_revision="$(git -C "$repository" rev-parse HEAD)"
  if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo \
      "$name revision mismatch: expected=$expected_revision actual=$actual_revision" \
      >&2
    exit 3
  fi
  if [[ -n "$(git -C "$repository" status --porcelain)" ]]; then
    echo "$name worktree is dirty" >&2
    exit 4
  fi
}

final_decision_ready() {
  "$python_bin" - \
    "$final_decision" \
    "$FIELDSCOPE_EXPECTED_SOURCE_REVISION" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
allowed = {
    "supports_core_hypothesis_with_scaling_extension",
    "supports_core_hypothesis_limited_scaling",
    "main_task_gain_not_causally_attributed",
    "limited_or_negative",
}
if (
    payload.get("verdict") not in allowed
    or payload.get("status") == "incomplete"
    or payload.get("code_revision") != revision
    or payload.get("code_dirty") is not False
):
    raise SystemExit(1)
PY
}

while true; do
  verify_checkout \
    "$source_repository" \
    "$FIELDSCOPE_EXPECTED_SOURCE_REVISION" \
    source
  verify_checkout \
    "$analyzer_repository" \
    "$FIELDSCOPE_EXPECTED_ANALYZER_REVISION" \
    analyzer
  if final_decision_ready; then
    break
  fi
  echo \
    "$(date --iso-8601=seconds) waiting for complete formal final decision $final_decision"
  sleep "$poll_seconds" 9>&-
done

echo "$(date --iso-8601=seconds) formal final decision complete; starting error analysis"
for dataset in imagenet100 voc2012 ade20k nyuv2; do
  verify_checkout \
    "$source_repository" \
    "$FIELDSCOPE_EXPECTED_SOURCE_REVISION" \
    source
  verify_checkout \
    "$analyzer_repository" \
    "$FIELDSCOPE_EXPECTED_ANALYZER_REVISION" \
    analyzer
  matrix="$source_repository/outputs/full_validation/$cache_tag/$dataset/matrix_report.json"
  task_output="$output_root/$dataset"
  echo "$(date --iso-8601=seconds) running supervised error analysis dataset=$dataset"
  (
    cd "$analyzer_repository"
    "$python_bin" -m fieldscope.cli run-readout-error-analysis \
      --matrix-report "$matrix" \
      --source-repository-root "$source_repository" \
      --output-dir "$task_output" \
      >/dev/null
  )
  "$python_bin" - "$task_output/summary.json" "$FIELDSCOPE_EXPECTED_SOURCE_REVISION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    payload.get("status") != "passed"
    or payload.get("changes_main_verdict") is not False
    or payload.get("formal_source", {}).get("code_revision") != sys.argv[2]
    or len(payload.get("input_reports", [])) != 33
    or len(payload.get("comparisons", {})) != 12
):
    raise SystemExit("supervised error-analysis summary contract mismatch")
PY
done

"$python_bin" - \
  "$output_root" \
  "$final_decision" \
  "$FIELDSCOPE_EXPECTED_SOURCE_REVISION" <<'PY'
import json
import pathlib
import sys

from fieldscope.experiments import atomic_json_dump, code_provenance, file_sha256

output_root = pathlib.Path(sys.argv[1]).resolve()
final_decision = pathlib.Path(sys.argv[2]).resolve()
source_revision = sys.argv[3]
tasks = {}
for dataset in ("imagenet100", "voc2012", "ade20k", "nyuv2"):
    path = output_root / dataset / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise SystemExit(f"incomplete supervised error analysis: {dataset}")
    tasks[dataset] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "task": payload.get("task"),
        "num_samples": payload.get("num_samples"),
        "sample_ids_sha256": payload.get("sample_ids_sha256"),
    }
atomic_json_dump(
    output_root / "registry.json",
    {
        "schema_version": 1,
        "status": "passed",
        "evidence_scope": "prospective_secondary_supervised_error_analysis",
        "changes_main_verdict": False,
        "formal_source_revision": source_revision,
        "formal_final_decision": {
            "path": str(final_decision),
            "sha256": file_sha256(final_decision),
        },
        "tasks": tasks,
        "analyzer": code_provenance(),
    },
)
PY

completion_audit="$output_root/completion_audit.json"
(
  cd "$analyzer_repository"
  "$python_bin" -m fieldscope.cli audit-research-completion \
    --source-repository-root "$source_repository" \
    --final-decision "$final_decision" \
    --error-registry "$output_root/registry.json" \
    --output "$completion_audit" \
    >/dev/null
)
"$python_bin" - "$completion_audit" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    payload.get("status") != "passed"
    or payload.get("execution_complete") is not True
    or payload.get("changes_scientific_verdict") is not False
):
    raise SystemExit("research completion audit contract mismatch")
PY
echo \
  "$(date --iso-8601=seconds) supervised error analysis and completion audit complete audit=$completion_audit"
