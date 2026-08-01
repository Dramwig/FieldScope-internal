#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"
python_bin="${FIELDSCOPE_PYTHON:-python}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
main_evidence="outputs/full_validation/$cache_tag/evidence_decision.json"
causal_evidence="outputs/full_validation/$cache_tag/causal/final_evidence_decision.json"
extension_evidence="outputs/full_validation/$cache_tag/extension/extension_evidence_decision.json"
final_evidence="outputs/full_validation/$cache_tag/final_evidence_decision.json"

bash scripts/eval/run_causal_validation_after_main.sh
main_verdict="$(
  "$python_bin" - "$main_evidence" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["verdict"])
PY
)"
extension_arguments=()
if [[ "$main_verdict" == "main_tasks_supported_pending_causal_audits" ]]; then
  bash scripts/eval/run_extension_after_main.sh
  extension_arguments=(--extension-evidence "$extension_evidence")
elif [[ "$main_verdict" != "limited_or_negative" ]]; then
  echo "unexpected main evidence verdict=$main_verdict" >&2
  exit 6
fi

"$python_bin" -m fieldscope.cli audit-final-evidence \
  --main-evidence "$main_evidence" \
  --causal-evidence "$causal_evidence" \
  "${extension_arguments[@]}" \
  --output "$final_evidence"
