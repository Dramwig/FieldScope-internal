#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 ROOT REPOSITORY REGISTRY DATASET WATCHDOG_SCRIPT OUTPUT" >&2
  exit 2
fi

root="$1"
repository="$2"
registry="$3"
dataset="$4"
watchdog_script="$5"
output="$6"
python_bin="$root/.venv/bin/python"
watchdog_pid_file="$root/recovery/orchestration/020c1de/watchdog-h200/watchdog.pid"
expected_revision="020c1de567edd88e0eda245fd085335ffe678f47"
poll_seconds="${FIELDSCOPE_PARALLEL_MONITOR_POLL_SECONDS:-30}"
watchdog_grace_seconds="${FIELDSCOPE_PARALLEL_MONITOR_WATCHDOG_GRACE_SECONDS:-900}"

if ! [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid poll seconds: $poll_seconds" >&2
  exit 2
fi
if ! [[ "$watchdog_grace_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid watchdog grace seconds: $watchdog_grace_seconds" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"

worker_identity() {
  "$python_bin" - "$registry" "$dataset" "$repository" "$expected_revision" <<'PY'
import json
import os
import pathlib
import sys

registry_path = pathlib.Path(sys.argv[1]).resolve()
dataset = sys.argv[2]
repository = pathlib.Path(sys.argv[3]).resolve()
revision = sys.argv[4]
payload = json.loads(registry_path.read_text(encoding="utf-8"))
if payload.get("status") != "active":
    raise SystemExit(1)
if payload.get("fixed_revision") != revision:
    raise SystemExit(1)
if pathlib.Path(payload.get("repository", "")).resolve() != repository:
    raise SystemExit(1)
entry = payload.get("workers", {}).get(dataset)
if not isinstance(entry, dict):
    raise SystemExit(1)
pid = int(entry["worker_pid"])
proc = pathlib.Path(f"/proc/{pid}")
if not proc.is_dir():
    raise SystemExit(1)
fields = (proc / "stat").read_text().split()
if int(fields[4]) != int(entry["pgid"]):
    raise SystemExit(1)
if int(fields[21]) != int(entry["worker_start_time_ticks"]):
    raise SystemExit(1)
if (proc / "cwd").resolve() != repository:
    raise SystemExit(1)
argv = (proc / "cmdline").read_bytes().split(b"\0")
argv = [item.decode(errors="replace") for item in argv if item]
if len(argv) < 3 or argv[1:3] != ["-m", "fieldscope.cli"]:
    raise SystemExit(1)
if "run-readout-matrix" not in argv:
    raise SystemExit(1)
if entry["expected_output_dir"] not in argv:
    raise SystemExit(1)
if not any(value.endswith(entry["expected_train_cache_suffix"]) for value in argv):
    raise SystemExit(1)
print(pid, int(entry["pgid"]), int(entry["worker_start_time_ticks"]))
PY
}

watchdog_alive() {
  [[ -f "$watchdog_pid_file" ]] || return 1
  local pid cwd
  read -r pid <"$watchdog_pid_file"
  [[ "$pid" =~ ^[1-9][0-9]*$ && -d "/proc/$pid" ]] || return 1
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ "$cwd" == "$root/recovery/orchestration/020c1de" ]] || return 1
  "$python_bin" - "$pid" "$watchdog_script" <<'PY'
import pathlib
import sys

pid = int(sys.argv[1])
expected = str(pathlib.Path(sys.argv[2]).resolve())
argv = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
argv = [item.decode(errors="replace") for item in argv if item]
cwd = pathlib.Path(f"/proc/{pid}/cwd").resolve()
script = pathlib.Path(argv[1]) if len(argv) == 2 else pathlib.Path()
actual = str((script if script.is_absolute() else cwd / script).resolve())
if len(argv) != 2 or argv[0] != "bash" or actual != expected:
    raise SystemExit(1)
PY
}

checkout_ready() {
  [[ "$(git -C "$repository" rev-parse HEAD 2>/dev/null || true)" == "$expected_revision" ]] &&
    [[ -z "$(git -C "$repository" status --porcelain 2>/dev/null)" ]]
}

terminate_group() {
  local pgid="$1"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep 15
  if worker_identity >/dev/null 2>&1; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}

if [[ "${FIELDSCOPE_PARALLEL_MONITOR_SELFTEST:-0}" == "1" ]]; then
  watchdog_alive
  checkout_ready
  echo "parallel monitor self-test passed"
  exit 0
fi

read -r worker_pid pgid worker_start_time < <(worker_identity)
missing_since=0
status="worker_exited"
reason="registered worker exited normally or failed independently"

while worker_identity >/dev/null 2>&1; do
  now="$(date +%s)"
  if ! checkout_ready; then
    status="terminated_checkout_invalid"
    reason="fixed-revision formal checkout became invalid"
    terminate_group "$pgid"
    break
  fi
  if watchdog_alive; then
    missing_since=0
  else
    if (( missing_since == 0 )); then
      missing_since="$now"
    elif (( now - missing_since >= watchdog_grace_seconds )); then
      status="terminated_watchdog_absent"
      reason="authoritative watchdog absent beyond grace period"
      terminate_group "$pgid"
      break
    fi
  fi
  sleep "$poll_seconds"
done

temporary="${output}.tmp.$$"
"$python_bin" - \
  "$temporary" "$status" "$reason" "$dataset" "$worker_pid" "$pgid" \
  "$worker_start_time" "$registry" "$expected_revision" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
registry = pathlib.Path(sys.argv[8]).resolve()
payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "reason": sys.argv[3],
    "dataset": sys.argv[4],
    "worker_pid": int(sys.argv[5]),
    "pgid": int(sys.argv[6]),
    "worker_start_time_ticks": int(sys.argv[7]),
    "registry": str(registry),
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "fixed_revision": sys.argv[9],
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "execution_complete": False,
    "method_effectiveness_conclusion": None,
    "changes_scientific_verdict": False,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv "$temporary" "$output"
