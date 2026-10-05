#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 ROOT REPOSITORY REGISTRY DATASET WATCHDOG_STATE OUTPUT" >&2
  exit 2
fi

root="$1"
repository="$2"
registry="$3"
dataset="$4"
watchdog_state="$5"
output="$6"
python_bin="$root/.venv/bin/python"
state_helper="${FIELDSCOPE_CROSS_HOST_STATE_HELPER:-$root/FieldScope-internal/scripts/ops/fieldscope_cross_host_monitor_state.py}"
expected_revision="020c1de567edd88e0eda245fd085335ffe678f47"
poll_seconds="${FIELDSCOPE_CROSS_HOST_MONITOR_POLL_SECONDS:-30}"
watchdog_grace_seconds="${FIELDSCOPE_CROSS_HOST_MONITOR_WATCHDOG_GRACE_SECONDS:-1200}"

for value in "$poll_seconds" "$watchdog_grace_seconds"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "poll and grace seconds must be positive integers" >&2
    exit 2
  fi
done
[[ -f "$state_helper" ]] || { echo "missing state helper: $state_helper" >&2; exit 2; }
mkdir -p "$(dirname "$output")"

worker_identity() {
  "$python_bin" - "$registry" "$dataset" "$repository" "$expected_revision" <<'PY'
import fcntl
import json
import os
import pathlib
import socket
import sys

registry_path = pathlib.Path(sys.argv[1]).resolve()
dataset = sys.argv[2]
repository = pathlib.Path(sys.argv[3]).resolve()
revision = sys.argv[4]
payload = json.loads(registry_path.read_text(encoding="utf-8"))
if payload.get("status") != "active" or payload.get("fixed_revision") != revision:
    raise SystemExit(1)
if pathlib.Path(payload.get("repository", "")).resolve() != repository:
    raise SystemExit(1)
entry = payload.get("workers", {}).get(dataset)
if not isinstance(entry, dict) or entry.get("runtime_hostname") != socket.gethostname():
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
argv = [item.decode(errors="replace") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
if len(argv) < 4 or argv[1:4] != ["-m", "fieldscope.cli", "run-readout-matrix"]:
    raise SystemExit(1)
if entry["expected_output_dir"] not in argv:
    raise SystemExit(1)
if not any(value.endswith(entry["expected_train_cache_suffix"]) for value in argv):
    raise SystemExit(1)
environment = {}
for item in (proc / "environ").read_bytes().split(b"\0"):
    if b"=" in item:
        key, value = item.split(b"=", 1)
        environment[key.decode(errors="replace")] = value.decode(errors="replace")
if environment.get("CUDA_VISIBLE_DEVICES") != str(entry["expected_cuda_visible_devices"]):
    raise SystemExit(1)
lock_path = pathlib.Path(entry["expected_lock_file"]).resolve()
if not lock_path.is_file():
    raise SystemExit(1)
with lock_path.open("r", encoding="utf-8") as handle:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)
        raise SystemExit(1)
print(pid, int(entry["pgid"]), int(entry["worker_start_time_ticks"]))
PY
}

watchdog_alive() {
  "$python_bin" "$state_helper" \
    --state "$watchdog_state" \
    --expected-revision "$expected_revision" \
    --max-age-seconds "$watchdog_grace_seconds" >/dev/null
}

checkout_ready() {
  [[ "$(git -C "$repository" rev-parse HEAD 2>/dev/null || true)" == "$expected_revision" ]] &&
    [[ -z "$(git -C "$repository" status --porcelain 2>/dev/null)" ]]
}

registered_worker_process_alive() {
  "$python_bin" - "$worker_pid" "$worker_start_time" <<'PY'
import pathlib
import sys

pid = int(sys.argv[1])
expected_start_time = int(sys.argv[2])
try:
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
except (FileNotFoundError, PermissionError, ProcessLookupError):
    raise SystemExit(1)
raise SystemExit(0 if int(fields[21]) == expected_start_time else 1)
PY
}

terminate_group() {
  local pgid="$1"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep 15
  if registered_worker_process_alive; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}

if [[ "${FIELDSCOPE_CROSS_HOST_MONITOR_SELFTEST:-0}" == "1" ]]; then
  worker_identity >/dev/null
  watchdog_alive
  checkout_ready
  echo "cross-host monitor self-test passed"
  exit 0
fi

read -r worker_pid pgid worker_start_time < <(worker_identity)
missing_since=0
status="worker_exited"
reason="registered worker exited normally or failed independently"

while registered_worker_process_alive; do
  now="$(date +%s)"
  if ! worker_identity >/dev/null 2>&1; then
    status="terminated_worker_identity_invalid"
    reason="registered worker remained alive but its identity or ownership lock became invalid"
    terminate_group "$pgid"
    break
  fi
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
      status="terminated_watchdog_heartbeat_absent"
      reason="shared authoritative watchdog heartbeat absent beyond grace period"
      terminate_group "$pgid"
      break
    fi
  fi
  sleep "$poll_seconds"
done

temporary="${output}.tmp.$$"
"$python_bin" - \
  "$temporary" "$status" "$reason" "$dataset" "$worker_pid" "$pgid" \
  "$worker_start_time" "$registry" "$watchdog_state" "$expected_revision" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
registry = pathlib.Path(sys.argv[8]).resolve()
watchdog_state = pathlib.Path(sys.argv[9]).resolve()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "reason": sys.argv[3],
    "dataset": sys.argv[4],
    "worker_pid": int(sys.argv[5]),
    "pgid": int(sys.argv[6]),
    "worker_start_time_ticks": int(sys.argv[7]),
    "registry": str(registry),
    "registry_sha256": sha256(registry),
    "watchdog_state": str(watchdog_state),
    "watchdog_state_sha256": sha256(watchdog_state),
    "fixed_revision": sys.argv[10],
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "execution_complete": False,
    "method_effectiveness_conclusion": None,
    "changes_scientific_verdict": False,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv "$temporary" "$output"
