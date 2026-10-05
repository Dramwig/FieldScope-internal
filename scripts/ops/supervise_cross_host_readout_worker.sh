#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 11 ]]; then
  echo "usage: $0 ROOT REPOSITORY DATASET GPU LOCK LAUNCHER REGISTRAR MONITOR REGISTRY WATCHDOG_STATE LOG_DIR" >&2
  exit 2
fi

root="$1"
repository="$2"
dataset="$3"
gpu="$4"
lock_file="$5"
launcher="$6"
registrar="$7"
monitor="$8"
registry="$9"
watchdog_state="${10}"
log_dir="${11}"
python_bin="$root/.venv/bin/python"
revision="020c1de567edd88e0eda245fd085335ffe678f47"
poll_seconds="${FIELDSCOPE_CROSS_HOST_SUPERVISOR_POLL_SECONDS:-60}"
restart_grace_seconds="${FIELDSCOPE_CROSS_HOST_SUPERVISOR_RESTART_GRACE_SECONDS:-180}"
supervisor_lock="${FIELDSCOPE_CROSS_HOST_SUPERVISOR_LOCK:-$root/recovery/orchestration/020c1de/cross-host-a800/${dataset}.supervisor.lock}"
state_helper="${FIELDSCOPE_CROSS_HOST_STATE_HELPER:-$root/FieldScope-internal/scripts/ops/fieldscope_cross_host_monitor_state.py}"
binding_pid_file="$root/recovery/orchestration/020c1de/cross-host-a800/${dataset}.supervised.monitor.pid"
binding_worker_file="$root/recovery/orchestration/020c1de/cross-host-a800/${dataset}.supervised.worker.pid"

[[ "$dataset" == nyuv2 || "$dataset" == voc2012 ]] || exit 2
[[ "$gpu" =~ ^[01]$ ]] || exit 2
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ && "$restart_grace_seconds" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$log_dir" "$(dirname "$supervisor_lock")"
exec 8>"$supervisor_lock"
flock -n 8 || exit 13

log() { printf '%s dataset=%s %s\n' "$(date --iso-8601=seconds)" "$dataset" "$*" >>"$log_dir/${dataset}.supervisor.log"; }

matrix_complete() {
  "$python_bin" - "$repository/outputs/full_validation/auraflow_v03/$dataset/matrix_report.json" <<'PY'
import json
import pathlib
import sys
try:
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
runs = payload.get("runs")
raise SystemExit(0 if payload.get("status") == "passed" and isinstance(runs, list) and len(runs) == 60 else 1)
PY
}

worker_identity() {
  "$python_bin" - "$repository" "$python_bin" "$dataset" "$gpu" <<'PY'
import os
import pathlib
import sys

repository = pathlib.Path(sys.argv[1]).resolve()
python_bin = pathlib.Path(sys.argv[2]).resolve()
dataset = sys.argv[3]
gpu = sys.argv[4]
output = f"outputs/full_validation/auraflow_v03/{dataset}"
cache = f"/feature_cache/auraflow_v03/{dataset}_train"
found = []
for proc in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        argv = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        if len(argv) < 4 or argv[1:4] != ["-m", "fieldscope.cli", "run-readout-matrix"]:
            continue
        if pathlib.Path(argv[0]).resolve() != python_bin or output not in argv:
            continue
        if not any(x.endswith(cache) for x in argv) or (proc / "cwd").resolve() != repository:
            continue
        env = {}
        for item in (proc / "environ").read_bytes().split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                env[key.decode(errors="replace")] = value.decode(errors="replace")
        if env.get("CUDA_VISIBLE_DEVICES") != gpu:
            continue
        found.append(int(proc.name))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
if len(found) == 1:
    print(found[0])
    raise SystemExit(0)
raise SystemExit(1)
PY
}

heartbeat_ok() {
  "$python_bin" "$state_helper" --state "$watchdog_state" --expected-revision "$revision" --max-age-seconds 1200 >/dev/null
}

checkout_ok() {
  [[ "$(git -C "$repository" rev-parse HEAD 2>/dev/null || true)" == "$revision" ]] &&
    [[ -z "$(git -C "$repository" status --porcelain 2>/dev/null)" ]]
}

lock_free() {
  exec 9>"$lock_file"
  if flock -n 9; then
    flock -u 9
    return 0
  fi
  return 1
}

monitor_binding_alive() {
  local expected_worker monitor_pid bound_worker
  expected_worker="$1"
  [[ -f "$binding_pid_file" && -f "$binding_worker_file" ]] || return 1
  read -r monitor_pid <"$binding_pid_file"
  read -r bound_worker <"$binding_worker_file"
  [[ "$bound_worker" == "$expected_worker" && "$monitor_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -d "/proc/$monitor_pid" ]] || return 1
  "$python_bin" - "$monitor_pid" "$monitor" "$dataset" "$registry" <<'PY'
import pathlib
import sys

pid = int(sys.argv[1])
expected_script = pathlib.Path(sys.argv[2]).resolve()
dataset = sys.argv[3]
registry = pathlib.Path(sys.argv[4]).resolve()
try:
    argv = [x.decode(errors="replace") for x in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
    cwd = pathlib.Path(f"/proc/{pid}/cwd").resolve()
except (FileNotFoundError, PermissionError, ProcessLookupError):
    raise SystemExit(1)
if len(argv) < 7 or pathlib.Path(argv[1] if argv[0] == "bash" else "").resolve() != expected_script:
    raise SystemExit(1)
if dataset not in argv or str(registry) not in argv:
    raise SystemExit(1)
raise SystemExit(0 if cwd.is_dir() else 1)
PY
}

register_workers() {
  "$python_bin" "$registrar" --root "$root" --repository "$repository" --output "$registry" \
    --worker "nyuv2:0:$root/logs/gpu67-parallel-20260823/nyuv2-gpu6.lock" \
    --worker "voc2012:1:$root/logs/gpu67-parallel-20260822/voc2012-gpu7.lock"
}

ensure_monitor() {
  local worker_pid monitor_pid stamp
  worker_pid="$1"
  if monitor_binding_alive "$worker_pid"; then
    return 0
  fi
  register_workers >/dev/null || return 1
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  nohup setsid bash "$monitor" "$root" "$repository" "$registry" "$dataset" "$watchdog_state" \
    "$root/recovery/orchestration/020c1de/cross-host-a800/${dataset}.monitor-${stamp}.exit.json" \
    </dev/null >>"$log_dir/${dataset}.supervisor.log" 2>&1 &
  monitor_pid="$!"
  printf '%s\n' "$monitor_pid" >"${binding_pid_file}.tmp.$$"
  mv "${binding_pid_file}.tmp.$$" "$binding_pid_file"
  printf '%s\n' "$worker_pid" >"${binding_worker_file}.tmp.$$"
  mv "${binding_worker_file}.tmp.$$" "$binding_worker_file"
  log "monitor_started pid=$monitor_pid worker_pid=$worker_pid stamp=$stamp"
}

start_worker_and_monitor() {
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log "starting recovery from last accepted checkpoint stamp=$stamp"
  nohup setsid bash "$launcher" "$root" "$repository" "$dataset" "$gpu" "$lock_file" \
    "$root/recovery/orchestration/020c1de/cross-host-a800/${dataset}.worker.pid" \
    "$log_dir/${dataset}.recovery-${stamp}.log" "$watchdog_state" \
    </dev/null >>"$log_dir/${dataset}.supervisor.log" 2>&1 &
  for _ in $(seq 1 30); do
    if worker_pid="$(worker_identity 2>/dev/null)"; then
      log "worker_started pid=$worker_pid"
      ensure_monitor "$worker_pid" || return 1
      return 0
    fi
    sleep 2
  done
  log "worker_start_timeout"
  return 1
}

if [[ "${FIELDSCOPE_CROSS_HOST_SUPERVISOR_SELFTEST:-0}" == "1" ]]; then
  heartbeat_ok
  checkout_ok
  ! matrix_complete
  worker_identity >/dev/null
  ! lock_free
  echo "cross-host supervisor self-test passed dataset=$dataset"
  exit 0
fi

missing_since=0
while true; do
  if matrix_complete; then
    log "matrix_complete supervisor_exiting"
    exit 0
  fi
  if worker_pid="$(worker_identity 2>/dev/null)"; then
    ensure_monitor "$worker_pid" || true
    missing_since=0
    sleep "$poll_seconds"
    continue
  fi
  if ! lock_free || ! heartbeat_ok || ! checkout_ok; then
    missing_since=0
    sleep "$poll_seconds"
    continue
  fi
  now="$(date +%s)"
  if (( missing_since == 0 )); then
    missing_since="$now"
  elif (( now - missing_since >= restart_grace_seconds )); then
    if start_worker_and_monitor; then
      missing_since=0
    else
      log "recovery_attempt_failed"
      missing_since="$now"
    fi
  fi
  sleep "$poll_seconds"
done
