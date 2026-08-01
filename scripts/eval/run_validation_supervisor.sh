#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"
: "${FIELDSCOPE_IMAGENET1K_ROOT:?Set FIELDSCOPE_IMAGENET1K_ROOT}"

repository="$FIELDSCOPE_ROOT/FieldScope-internal"
python_bin="${FIELDSCOPE_PYTHON:-$FIELDSCOPE_ROOT/.venv/bin/python}"
log_root="${FIELDSCOPE_LOG_ROOT:-$FIELDSCOPE_ROOT/logs}"
tag="${FIELDSCOPE_SUPERVISOR_TAG:-${FIELDSCOPE_EXPECTED_REVISION:0:7}}"
signal_log="$log_root/signal_gate_${tag}.log"
full_log="$log_root/full_validation_${tag}.log"
supervisor_log="$log_root/full_validation_supervisor_${tag}.log"
state_file="$log_root/full_validation_supervisor_${tag}.state"
pid_file="$log_root/full_validation_supervisor_${tag}.pid"
lock_file="$log_root/full_validation_supervisor_${tag}.lock"
signal_decision="$repository/outputs/signal_gate/promotion_decision.json"
final_decision="$repository/outputs/full_validation/auraflow_v03/final_evidence_decision.json"
external_wait_status="${FIELDSCOPE_EXTERNAL_WAIT_STATUS:-/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_300k_ema_teacher/reports/readiness_waiter.json}"
max_restarts="${FIELDSCOPE_MAX_RESTARTS:-5}"
poll_seconds="${FIELDSCOPE_SUPERVISOR_POLL_SECONDS:-60}"

mkdir -p "$log_root"
touch "$supervisor_log"
exec 9>"$lock_file"
if ! flock -n 9; then
  exit 0
fi
printf '%s\n' "$$" >"${pid_file}.tmp.$$"
mv -f "${pid_file}.tmp.$$" "$pid_file"
trap 'rm -f "$pid_file"' EXIT

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$supervisor_log"
}

load_state() {
  signal_restarts=0
  full_restarts=0
  if [[ -f "$state_file" ]]; then
    # shellcheck disable=SC1090
    source "$state_file"
  fi
  : "${signal_restarts:=0}"
  : "${full_restarts:=0}"
}

save_state() {
  local temporary="${state_file}.tmp.$$"
  printf 'signal_restarts=%q\nfull_restarts=%q\n' \
    "$signal_restarts" "$full_restarts" >"$temporary"
  mv -f "$temporary" "$state_file"
}

verify_repository() {
  local actual
  actual="$(git -C "$repository" rev-parse HEAD)"
  if [[ "$actual" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
    log "fatal revision mismatch expected=$FIELDSCOPE_EXPECTED_REVISION actual=$actual"
    return 1
  fi
  if [[ -n "$(git -C "$repository" status --porcelain)" ]]; then
    log "fatal dirty worktree"
    return 1
  fi
}

valid_decision() {
  local path="$1"
  local kind="$2"
  "$python_bin" - "$path" "$kind" "$FIELDSCOPE_EXPECTED_REVISION" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
kind = sys.argv[2]
revision = sys.argv[3]
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("code_revision") != revision or payload.get("code_dirty") is not False:
    raise SystemExit(1)
if kind == "signal":
    allowed = {"proceed", "stop_or_redesign"}
else:
    allowed = {
        "supports_core_hypothesis_with_scaling_extension",
        "supports_core_hypothesis_limited_scaling",
        "main_task_gain_not_causally_attributed",
        "limited_or_negative",
    }
if payload.get("verdict") not in allowed or payload.get("status") == "incomplete":
    raise SystemExit(1)
PY
}

find_exact_bash_script() {
  local expected_script="$1"
  local process_dir pid cwd
  local -a argv
  for process_dir in /proc/[0-9]*; do
    pid="${process_dir##*/}"
    [[ "$pid" != "$$" ]] || continue
    cwd="$(readlink -f "$process_dir/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$repository" ]] || continue
    argv=()
    mapfile -d '' -t argv <"$process_dir/cmdline" 2>/dev/null || continue
    if [[ ${#argv[@]} -eq 2 && \
          "${argv[0]}" == "bash" && \
          "${argv[1]}" == "$expected_script" ]]; then
      printf '%s\n' "$pid"
      return 0
    fi
  done
}

find_signal_pid() {
  find_exact_bash_script "scripts/eval/run_signal_gate_after_gpu.sh"
}

find_full_pid() {
  local script pid
  for script in \
    scripts/eval/run_full_validation_when_ready.sh \
    scripts/eval/run_full_validation_after_signal_gate.sh \
    scripts/eval/run_final_conclusion_after_main.sh \
    scripts/eval/run_causal_validation_after_main.sh \
    scripts/eval/run_extension_after_main.sh; do
    pid="$(find_exact_bash_script "$script")"
    if [[ -n "$pid" ]]; then
      printf '%s\n' "$pid"
      return 0
    fi
  done
}

active_child_workload() {
  local process_dir pid cwd
  local -a argv
  for process_dir in /proc/[0-9]*; do
    pid="${process_dir##*/}"
    [[ "$pid" != "$$" ]] || continue
    cwd="$(readlink -f "$process_dir/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$repository" ]] || continue
    argv=()
    mapfile -d '' -t argv <"$process_dir/cmdline" 2>/dev/null || continue
    if [[ ${#argv[@]} -ge 3 && \
          "${argv[0]}" == "$python_bin" && \
          "${argv[1]}" == "-m" && \
          "${argv[2]}" == "fieldscope.cli" ]]; then
      return 0
    fi
    if [[ ${#argv[@]} -ge 2 && "${argv[0]}" == "bash" ]]; then
      case "${argv[1]}" in
        scripts/eval/run_signal_gate_after_gpu.sh | \
        scripts/eval/run_full_validation_when_ready.sh | \
        scripts/eval/run_full_validation_after_signal_gate.sh | \
        scripts/eval/run_final_conclusion_after_main.sh | \
        scripts/eval/run_causal_validation_after_main.sh | \
        scripts/eval/run_extension_after_main.sh)
          ;;
        scripts/eval/*.sh | scripts/train/*.sh | scripts/data/*.sh)
          return 0
          ;;
      esac
    fi
  done
  return 1
}

external_wait_target() {
  local process_dir pid index
  local -a argv
  for process_dir in /proc/[0-9]*; do
    pid="${process_dir##*/}"
    argv=()
    mapfile -d '' -t argv <"$process_dir/cmdline" 2>/dev/null || continue
    if [[ ${#argv[@]} -lt 4 || \
          "${argv[1]}" != "scripts/run_generation_stability_full_readiness_waiter.py" ]]; then
      continue
    fi
    for ((index = 2; index + 1 < ${#argv[@]}; index++)); do
      if [[ "${argv[index]}" == "--status-output" && \
            "${argv[index + 1]}" == "$external_wait_status" ]]; then
        printf '%s\n' "$pid"
        return 0
      fi
    done
  done
}

launch_signal() {
  if (( signal_restarts >= max_restarts )); then
    log "fatal signal restart limit reached count=$signal_restarts"
    return 1
  fi
  signal_restarts=$((signal_restarts + 1))
  save_state
  local pid wait_target
  wait_target="$(external_wait_target)"
  pid="$({
    cd "$repository"
    setsid env \
      FIELDSCOPE_ROOT="$FIELDSCOPE_ROOT" \
      FIELDSCOPE_DATASETS_ROOT="$FIELDSCOPE_ROOT/datasets" \
      FIELDSCOPE_CHECKPOINTS_ROOT="$FIELDSCOPE_ROOT/checkpoints" \
      HF_HOME="$FIELDSCOPE_ROOT/hf_home" \
      FIELDSCOPE_PYTHON="$python_bin" \
      FIELDSCOPE_EXPECTED_REVISION="$FIELDSCOPE_EXPECTED_REVISION" \
      FIELDSCOPE_WAIT_FOR_PID="$wait_target" \
      bash scripts/eval/run_signal_gate_after_gpu.sh \
      9>&- \
      >>"$signal_log" 2>&1 </dev/null &
    echo $!
  })"
  log "restarted signal pid=$pid wait_pid=${wait_target:-none} attempt=$signal_restarts/$max_restarts"
  printf '%s\n' "$pid"
}

launch_full() {
  local signal_pid="$1"
  if (( full_restarts >= max_restarts )); then
    log "fatal full restart limit reached count=$full_restarts"
    return 1
  fi
  full_restarts=$((full_restarts + 1))
  save_state
  local pid
  pid="$({
    cd "$repository"
    setsid env \
      FIELDSCOPE_ROOT="$FIELDSCOPE_ROOT" \
      FIELDSCOPE_DATASETS_ROOT="$FIELDSCOPE_ROOT/datasets" \
      FIELDSCOPE_CHECKPOINTS_ROOT="$FIELDSCOPE_ROOT/checkpoints" \
      FIELDSCOPE_IMAGENET1K_ROOT="$FIELDSCOPE_IMAGENET1K_ROOT" \
      HF_HOME="$FIELDSCOPE_ROOT/hf_home" \
      FIELDSCOPE_PYTHON="$python_bin" \
      FIELDSCOPE_EXPECTED_REVISION="$FIELDSCOPE_EXPECTED_REVISION" \
      FIELDSCOPE_SIGNAL_PID="$signal_pid" \
      bash scripts/eval/run_full_validation_when_ready.sh \
      9>&- \
      >>"$full_log" 2>&1 </dev/null &
    echo $!
  })"
  log "restarted full pid=$pid signal_pid=$signal_pid attempt=$full_restarts/$max_restarts"
}

load_state
log "supervisor started revision=$FIELDSCOPE_EXPECTED_REVISION signal_restarts=$signal_restarts full_restarts=$full_restarts"

while true; do
  verify_repository || exit 2
  if valid_decision "$final_decision" final; then
    log "complete final decision observed path=$final_decision"
    exit 0
  fi

  signal_pid="$(find_signal_pid || true)"
  full_pid="$(find_full_pid || true)"
  signal_complete=false
  if valid_decision "$signal_decision" signal; then
    signal_complete=true
  fi

  if [[ "$signal_complete" != true && -z "$signal_pid" ]]; then
    if active_child_workload; then
      log "signal top-level absent but child workload remains; deferring restart"
    else
      signal_pid="$(launch_signal)" || exit 3
      load_state
    fi
  fi

  if [[ -z "$full_pid" ]]; then
    if active_child_workload; then
      log "full top-level absent but child workload remains; deferring restart"
    else
      if [[ -z "$signal_pid" ]]; then
        signal_pid="999999999"
      fi
      launch_full "$signal_pid" || exit 4
    fi
  fi

  sleep "$poll_seconds" 9>&-
done
