#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_CHECKPOINTS_ROOT:?Set FIELDSCOPE_CHECKPOINTS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"
: "${FIELDSCOPE_SIGNAL_PID:?Set FIELDSCOPE_SIGNAL_PID}"

python_bin="${FIELDSCOPE_PYTHON:-python}"
archive="${FIELDSCOPE_IMAGENET100_ARCHIVE:-$FIELDSCOPE_DATASETS_ROOT/raw/imagenet100/imagenet100_full.tar}"
expected_archive_bytes=17319391232
decision="${FIELDSCOPE_SIGNAL_ROOT:-outputs/signal_gate}/promotion_decision.json"
asset_report="outputs/asset_verification/imagenet100_remote_asset_verification.json"
asset_audit="outputs/asset_verification/imagenet100_remote_split_audit.json"
stable_archive_checks=0
asset_ready=false
signal_ready=false

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
    echo "refusing to orchestrate full validation from a dirty worktree" >&2
    exit 4
  fi
}

verify_signal_process() {
  local pid="$1"
  local argv=()
  mapfile -d '' -t argv <"/proc/$pid/cmdline"
  if [[ ${#argv[@]} -ne 2 || ${argv[0]} != bash || ${argv[1]} != scripts/eval/run_signal_gate_after_gpu.sh ]]; then
    echo "unexpected signal-gate process identity for pid=$pid" >&2
    exit 5
  fi
  local revision
  revision="$(
    tr '\0' '\n' <"/proc/$pid/environ" |
      sed -n 's/^FIELDSCOPE_EXPECTED_REVISION=//p'
  )"
  if [[ "$revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
    echo \
      "signal-gate process revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$revision" \
      >&2
    exit 6
  fi
}

verify_revision
while [[ "$asset_ready" != true || "$signal_ready" != true ]]; do
  verify_revision
  if [[ "$asset_ready" != true ]]; then
    archive_bytes="$(stat -c '%s' "$archive" 2>/dev/null || printf '0')"
    if (( archive_bytes > expected_archive_bytes )); then
      echo \
        "ImageNet-100 archive exceeds expected size: expected=$expected_archive_bytes actual=$archive_bytes" \
        >&2
      exit 7
    fi
    if (( archive_bytes == expected_archive_bytes )); then
      stable_archive_checks=$((stable_archive_checks + 1))
      echo \
        "$(date --iso-8601=seconds) ImageNet-100 archive size stable check $stable_archive_checks/3"
    else
      stable_archive_checks=0
      echo \
        "$(date --iso-8601=seconds) waiting for ImageNet-100 archive bytes=$archive_bytes/$expected_archive_bytes"
    fi
    if (( stable_archive_checks >= 3 )); then
      bash scripts/data/verify_prepare_imagenet100_remote.sh
      "$python_bin" - "$asset_report" "$asset_audit" "$FIELDSCOPE_EXPECTED_REVISION" <<'PY'
import json
import sys

asset = json.load(open(sys.argv[1], encoding="utf-8"))
audit = json.load(open(sys.argv[2], encoding="utf-8"))
if asset.get("status") != "passed":
    raise SystemExit("ImageNet-100 asset verification did not pass")
if audit.get("status") != "passed":
    raise SystemExit("ImageNet-100 split audit did not pass")
if audit.get("code_revision") != sys.argv[3] or audit.get("code_dirty") is not False:
    raise SystemExit("ImageNet-100 split audit provenance mismatch")
PY
      asset_ready=true
      echo "$(date --iso-8601=seconds) ImageNet-100 asset gate passed"
    fi
  fi

  if [[ "$signal_ready" != true ]]; then
    if kill -0 "$FIELDSCOPE_SIGNAL_PID" 2>/dev/null; then
      verify_signal_process "$FIELDSCOPE_SIGNAL_PID"
      echo \
        "$(date --iso-8601=seconds) waiting for signal gate pid=$FIELDSCOPE_SIGNAL_PID"
    else
      if [[ ! -f "$decision" ]]; then
        echo "signal gate exited without decision: $decision" >&2
        exit 8
      fi
      read -r verdict revision < <(
        "$python_bin" - "$decision" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("verdict", ""), payload.get("code_revision", ""))
PY
      )
      if [[ "$revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
        echo \
          "signal decision revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$revision" \
          >&2
        exit 9
      fi
      if [[ "$verdict" != "proceed" ]]; then
        echo "formal validation not promoted: verdict=$verdict" >&2
        exit 10
      fi
      signal_ready=true
      echo "$(date --iso-8601=seconds) signal gate promoted formal validation"
    fi
  fi

  if [[ "$asset_ready" != true || "$signal_ready" != true ]]; then
    sleep 60
  fi
done

verify_revision
exec bash scripts/eval/run_full_validation_after_signal_gate.sh
