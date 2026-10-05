#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ROOT ROLE GATE A800_SUPERVISOR A800_MONITOR" >&2
  exit 2
fi

root="$1"
role="$2"
gate="$3"
a800_supervisor="$4"
a800_monitor="$5"
python_bin="$root/.venv/bin/python"
repository="$root/recovery/worktrees/formal-020c1de"
handback="$root/recovery/orchestration/020c1de/handback"
watcher_state="$root/logs/auto-strong-audits/state.json"
a800_registry="$root/recovery/orchestration/020c1de/cross-host-a800/registered_workers.json"
h200_registry="$root/recovery/orchestration/020c1de/parallel-monitor/gpu67_registered_parallel_workers-486c21301c656fedf2ea662a0240e2c14fe43708e195a2dd178ca6ded0aea189.json"
h200_monitor="$root/recovery/orchestration/020c1de/parallel-monitor/monitor_registered_parallel_worker-e0677f25e01dccb4ccaf723a7f188efecd5cc760aa423877bce5a843161b8960.sh"
recovery_wrapper="$root/recovery/orchestration/020c1de/wrappers/run_fixed_revision_readout_recovery_h200.sh"
ade_lock="$root/logs/gpu67-parallel-20260823/ade20k-gpu7.lock"
poll_seconds="${FIELDSCOPE_HANDBACK_POLL_SECONDS:-30}"
restart_seconds="${FIELDSCOPE_HANDBACK_RESTART_SECONDS:-60}"

[[ "$role" == h200 || "$role" == a800 ]] || exit 2
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$restart_seconds" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "$gate" && -x "$python_bin" ]] || exit 2

mkdir -p "$handback"
exec 8>"$handback/${role}.gate-supervisor.lock"
flock -n 8 || exit 13
printf '%s\n' "$$" >"$handback/.${role}.gate-supervisor.pid.tmp.$$"
mv "$handback/.${role}.gate-supervisor.pid.tmp.$$" "$handback/${role}.gate-supervisor.pid"
trap 'rm -f "$handback/$role.gate-supervisor.pid"' EXIT

log() {
  printf '%s role=%s %s\n' "$(date --iso-8601=seconds)" "$role" "$*" \
    >>"$handback/${role}.gate-supervisor.log"
}

json_status_is() {
  local path="$1" expected="$2"
  "$python_bin" - "$path" "$expected" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == sys.argv[2] else 1)
PY
}

run_a800() {
  "$python_bin" "$gate" run-a800 \
    --registration "$handback/a800-registration.json" \
    --repository "$repository" \
    --watcher-state "$watcher_state" \
    --a800-registry "$a800_registry" \
    --a800-supervisor "$a800_supervisor" \
    --a800-monitor "$a800_monitor" \
    --heartbeat "$handback/a800-heartbeat.json" \
    --request "$handback/release-request.json" \
    --ack "$handback/a800-ack.json" \
    --poll "$poll_seconds"
}

run_h200() {
  "$python_bin" "$gate" run-h200 \
    --registration "$handback/h200-registration.json" \
    --watcher-state "$watcher_state" \
    --responder-heartbeat "$handback/a800-heartbeat.json" \
    --heartbeat "$handback/h200-heartbeat.json" \
    --request "$handback/release-request.json" \
    --a800-ack "$handback/a800-ack.json" \
    --release-complete "$handback/release-complete.json" \
    --poll "$poll_seconds"
}

register_current() {
  if [[ "$role" == a800 ]]; then
    "$python_bin" "$gate" register-a800 \
      --repository "$repository" \
      --a800-registry "$a800_registry" \
      --a800-supervisor "$a800_supervisor" \
      --a800-monitor "$a800_monitor" \
      --output "$handback/a800-registration.json"
    return
  fi
  local recovery_pid
  recovery_pid="$({
    "$python_bin" - "$root/recovery/orchestration/020c1de/watchdog-h200/state.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
recovery = payload.get("recovery", {})
if recovery.get("alive") is not True or not isinstance(recovery.get("pid"), int):
    raise SystemExit(1)
print(recovery["pid"])
PY
  })"
  "$python_bin" "$gate" register-h200 \
    --repository "$repository" \
    --recovery-pid "$recovery_pid" \
    --recovery-wrapper "$recovery_wrapper" \
    --image-output-token outputs/full_validation/auraflow_v03/imagenet100 \
    --h200-registry "$h200_registry" \
    --h200-monitor "$h200_monitor" \
    --ade-lock "$ade_lock" \
    --output "$handback/h200-registration.json"
}

if [[ "${FIELDSCOPE_HANDBACK_SUPERVISOR_SELFTEST:-0}" == 1 ]]; then
  if [[ "$role" == a800 ]]; then
    "$python_bin" "$gate" audit-a800 \
      --registration "$handback/a800-registration.json" \
      --repository "$repository" \
      --watcher-state "$watcher_state" \
      --a800-registry "$a800_registry" \
      --a800-supervisor "$a800_supervisor" \
      --a800-monitor "$a800_monitor" \
      --heartbeat "$handback/a800-heartbeat.json" \
      --request "$handback/release-request.json" \
      --ack "$handback/a800-ack.json"
  else
    "$python_bin" "$gate" audit-h200 \
      --registration "$handback/h200-registration.json" \
      --watcher-state "$watcher_state" \
      --responder-heartbeat "$handback/a800-heartbeat.json" \
      --heartbeat "$handback/h200-heartbeat.json" \
      --request "$handback/release-request.json" \
      --a800-ack "$handback/a800-ack.json" \
      --release-complete "$handback/release-complete.json"
  fi
  echo "cross-lane handback supervisor self-test passed role=$role"
  exit 0
fi

while true; do
  if [[ "$role" == a800 ]] && json_status_is "$handback/a800-ack.json" released; then
    log "release acknowledged; supervisor exiting"
    exit 0
  fi
  if [[ "$role" == h200 ]] && json_status_is "$handback/release-complete.json" released; then
    # One final coordinator pass resumes an exactly identified stopped parent
    # after a crash between durable completion write and SIGCONT.
    if run_h200; then
      log "release complete; supervisor exiting"
      exit 0
    fi
  fi
  log "starting gate"
  if [[ ! -e "$handback/release-request.json" ]]; then
    if ! register_current >>"$handback/${role}.gate.log" 2>&1; then
      log "registration failed"
      sleep "$restart_seconds"
      continue
    fi
  fi
  set +e
  if [[ "$role" == a800 ]]; then
    run_a800 >>"$handback/${role}.gate.log" 2>&1
  else
    run_h200 >>"$handback/${role}.gate.log" 2>&1
  fi
  rc="$?"
  set -e
  log "gate exited rc=$rc"
  sleep "$restart_seconds"
done
